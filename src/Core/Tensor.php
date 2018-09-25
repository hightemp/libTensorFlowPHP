<?php

namespace libTensorFlowPHP\Core;

use Exception;
use libTensorFlowPHP\Console;
use libTensorFlowPHP\Core\Utilities;
use libTensorFlowPHP\Core\OpHandler;

class TensorData 
{
  public $dataId;
  public $values;
}

interface TensorTracker 
{
  function fnRegisterTensor($oT);
  function fnDisposeTensor($oT);
  function fnWrite($mDataId, $aValues);
  function fnRead($mDataId);
  function fnReadSync($mDataId);
  function fnRegisterVariable($oV);
}

class TensorBuffer 
{
  public $size;
  public $shape;
  public $strides;
  public $values;

  function __construct($aShape, $sDtype, $aValues) 
  {
    $this->shape = clone $aShape;
    $this->size = Utilities::fnSizeFromShape($aShape);

    if ($aValues != null) {
      $iN = count($aValues);
      Utilities::fnAssert(
        $iN === $this->size,
        self::fnFormatString(
          "Length of values '?:' does not match the size " .
            "inferred by the shape '?:'.",
          $iN,
          $this->size
        )
      );
    }
    if ($sDtype === 'complex64') {
      throw new Exception(
        "complex64 dtype TensorBuffers are not supported. Please create " .
        "a TensorBuffer for the real and imaginary parts separately and " .
        "call tf.complex(real, imag)."
      );
    }
    $this->values = $aValues ||
      Utilities::fnGetTypedArrayFromDType($sDtype, Utilities::fnSizeFromShape($this->shape));
    $this->strides = Utilities::fnComputeStrides($aShape);
  }
  
  public function fnSet($iValue, ...$aLocs)
  {
    if (count($aLocs) === 0) {
      $aLocs = [0];
    }
    Utilities::fnAssert(
      count($aLocs) === $this->fnRank(),
      self::fnFormatString(
        "The number of provided coordinates (?:) must " .
          "match the rank (?:)",
        count($aLocs),
        $this->fnRank()
      )  
    );

    $iIndex = $this->fnLocToIndex($aLocs);
    $this->values[$iIndex] = $iValue;
  }
  
  public function fnGet(...$aLocs)
  {
    if (count($aLocs) === 0) {
      $aLocs = [0];
    }
    $iIndex = $aLocs[count($aLocs) - 1];
    for ($iI = 0; $iI < count($aLocs) - 1; ++$iI) {
      $iIndex += $this->strides[$iI] * $aLocs[$iI];
    }
    return $this->values[$iIndex];
  }
  
  public function fnLocToIndex($aLocs)
  {
    if ($this->fnRank() === 0) {
      return 0;
    } else if ($this->fnRank() === 1) {
      return $aLocs[0];
    }
    $iIndex = $aLocs[count($aLocs) - 1];
    for ($iI = 0; $iI < count($aLocs) - 1; ++$iI) {
      $iIndex += $this->strides[$iI] * $aLocs[$iI];
    }
    return $iIndex;
  }
  
  public function fnIndexToLoc($iIndex)
  {
    if ($this->fnRank() === 0) {
      return [];
    } else if ($this->fnRank() === 1) {
      return [$iIndex];
    }
    $aLocs = array_fill(0, count($this->shape), 0);
    for ($iI = 0; $iI < count($aLocs) - 1; ++$iI) {
      $aLocs[$iI] = floor($iIndex / $this->strides[$iI]);
      $iIndex -= $aLocs[$iI] * $this->strides[$iI];
    }
    $aLocs[count($aLocs) - 1] = $iIndex;
    return $aLocs;
  }
  
  public function fnRank() 
  {
    return count($this->shape);
  }

  public function fnToTensor()
  {
    return Tensor::fnMake($this->shape, ['values' => $this->values], $this->dtype);
  }
}

class Tensor
{
  public static $nextId = 0;

  public $id;
  public $dataId;
  public $shape;
  public $size;
  public $dtype;
  public $rankType;
  public $strides;

  public static $fnTrackerFn = null;

  public static function fnSetTensorTracker($fnFn) {
    self::$fnTrackerFn = $fnFn;
  }
  
  function __construct($aShape, $sDtype, $aValues, $mDataId) 
  {
    $this->shape = clone $aShape;
    $this->dtype = $sDtype || 'float32';
    $this->size = Utilities::fnSizeFromShape($aShape);
    if ($aValues != null) {
      Utilities::fnAssert(
        $this->size === count($aValues),
        self::fnFormatString(
          "Based on the provided shape, [?:], and dtype " .
            "?:, the tensor should have " .
            "?: values but has ?:",
            $aShape,
            $this->dtype,
            $this->size,
            count($aValues)
        )
      );
    }

    $this->strides = computeStrides(shape);
    $this->dataId = $mDataId != null ? $mDataId : [];
    $this->id = self::$nextId++;
    $this->rankType = $this->fnRank() < 5 ? $this->fnRank() : 'higher';
    self::$fnTrackerFn()->fnRegisterTensor($this);
    if ($aValues != null) {
      self::$fnTrackerFn()->fnWrite($this->dataId, $aValues);
    }
  }
  
  public static function fnMake($aShape, $oData, $sDtype)
  {
    return new Tensor($aShape, $sDtype, $oData->values, $oData->dataId);
  }
  
  public function fnFlatten()
  {
    $this->fnThrowIfDisposed();
    return $this->fnAs1D();
  }
  
  public function fnAsScalar()
  {
    $this->fnThrowIfDisposed();
    Utilities::fnAssert($this->size === 1, 'The array must have only 1 element.');
    return $this->fnReshape([]);
  }

  public function fnAs1D()
  {
    $this->fnThrowIfDisposed();
    return $this->fnReshape([$this->size]);
  }
  
  public function fnAs2D($iRows, $iColumns)
  {
    $this->fnThrowIfDisposed();
    return $this->fnReshape([$iRows, $iColumns]);
  }

  public function fnAs3D($iRows, $iColumns, $iDepth)
  {
    $this->fnThrowIfDisposed();
    return $this->fnReshape([$iRows, $iColumns, $iDepth]);
  }

  public function fnAs4D($iRows, $iColumns, $iDepth, $iDepth2)
  {
    $this->fnThrowIfDisposed();
    return $this->fnReshape([$iRows, $iColumns, $iDepth, $iDepth2]);
  }

  public function fnAsType($sDtype)
  {
    $this->fnThrowIfDisposed();
    return OpHandler::$opHandler->fnCast($this, $sDtype);
  }
  
  public function fnRank()
  {
    return count($this->shape);
  }
}

