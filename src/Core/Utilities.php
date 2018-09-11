<?php

namespace libTensorFlowPHP\Core;

use Exception;

class Utilities
{
  public static function fnShuffle($aArray) 
  {
    $iCounter = count($aArray);
    $iTemp = 0;
    $iIndex = 0;
    // While there are elements in the array
    while ($iCounter > 0) {
      // Pick a random index
      $iIndex = mt_rand(0, $iCounter-1);
      // Decrease counter by 1
      $iCounter--;
      // And swap the last element with it
      $iTemp = $aArray[$iCounter];
      $aArray[$iCounter] = $aArray[$iIndex];
      $aArray[$iIndex] = $iTemp;
    }
  }
  
  public static function fnClamp($iMin, $iX, $iMax) 
  {
    return max([$iMin, min([$iX, $iMax])]);
  }
  
  public static function fnRandUniform($iA, $iB) 
  {
    $fR = mt_rand(0, 1000) / 1000;
    return ($iB * $fR) + (1 - $fR) * $iA;
  }
  
  public static function fnDistSquared($aA, $aB) 
  {
    $fResult = 0;
    for ($iI = 0; $iI < count($aA); $iI++) {
      $fDiff = $aA[$iI] - $aB[$iI];
      $fResult += $fDiff * $fDiff;
    }
    return $fResult;
  }
  
  public static function fnAssert($bExpr, $mMsg) 
  {
    if (!$bExpr) {
      throw new Exception(is_callable($mMsg)? $mMsg() : $mMsg);
    }
  }
  
  public static function fnAssertShapesMatch($aShapeA, $aShapeB, $sErrorMessagePrefix = '') 
  {
    self::fnAssert(
      self::fnArraysEqual($aShapeA, $aShapeB),
      self::fnFormatString($sErrorMessagePrefix + " Shapes ?: and ?: must match", $aShapeA, $aShapeB)
    );
  }
  
  public static function fnFormatString($sString, ...$aArguments)
  {
    return preg_replace_callback(
      "/\?:/", 
      function() use ($aArguments)
      {
        static $iCounter = 0;
        $sResult = json_encode($aArguments[$iCounter]);
        $iCounter++;
        return $sResult;
      },
      $sString
    );
  }
  
  public static function fnAssertNonNull($mA) 
  {
    self::fnAssert(
      $mA != null,
      "The input to the tensor constructor must be a non-null value."
    );
  }
  
  public static function fnFlatten($mArr, &$aRet = []) 
  {
    if (is_array($mArr)) {
      for ($iI = 0; $iI < count($mArr); ++$iI) {
        self::fnFlatten($mArr[$iI], $aRet);
      }
    } else {
      array_push($aRet, $mArr);
    }
    return $aRet;
  }
  
  public static function fnInferShape($mVal)
  {
    $mFirstElem = $mVal;

    //if (isTypedArray(val)) {
    //  return [(val as TypedArray).length];
    //}
    if (!is_array($mVal)) {
      return [];  // Scalar.
    }
    $aShape = [];

    while (is_array($mFirstElem)) {
      array_push($aShape, count($mFirstElem));
      $mFirstElem = $mFirstElem[0];
    }
    if (is_array($mVal)) {
      self::fnDeepAssertShapeConsistency($mVal, $aShape, []);
    }

    return $aShape;
  }
  
  public static function fnDeepAssertShapeConsistency($mVal, $aShape, $iIndices = []) 
  {
    $iIndices = (array) $iIndices;
    if (!is_array($mVal)) {
      self::fnAssert(
        count($aShape) === 0,
        self::fnFormatString(
          "Element arr[?:] is a primitive, but should be an array of ?: elements", 
          join('][', $iIndices), 
          $aShape[0]
        )
      );
      return;
    }
    self::fnAssert(
      count($aShape) > 0,
      self::fnFormatString(
        "Element arr[?:] should be a primitive, but is an array of ?: elements",
        join('][', $iIndices),
        count($mVal)
      )
    );
    self::fnAssert(
      count($mVal) === $aShape[0],
      self::fnFormatString(
        "`Element arr[?:] should have ?: elements, but has ?: elements",
        join('][', $iIndices),
        $aShape[0],
        count($mVal)
      )
    );
    $aSubShape = array_slice($aShape, 1);
    for ($iI = 0; $iI < count($mVal); ++$iI) {
      self::fnDeepAssertShapeConsistency($mVal[$iI], $aSubShape, array_merge($iIndices, [$iI]));
    }
  }
  
  public static function fnSizeFromShape($aShape) 
  {
    if (count($aShape) === 0) {
      // Scalar.
      return 1;
    }
    $iSize = $aShape[0];
    for ($iI = 1; $iI < count($aShape); $iI++) {
      $iSize *= $aShape[$iI];
    }
    return $iSize;
  }
  
  public static function fnIsScalarShape($aShape) 
  {
    return count($aShape) === 0;
  }
  
  public static function fnArraysEqual($aN1, $aN2) 
  {
    /*
    if (count($aN1) !== count($aN2)) {
      return false;
    }
    for (let i = 0; i < n1.length; i++) {
      if (n1[i] !== n2[i]) {
        return false;
      }
    }
    return true;
     */
    return empty(array_diff($aN1, $aN2));
  }
  
  public static function fnIsInt($iA) 
  {
    //return $iA % 1 === 0;
    return is_int($iA);
  }
  
  public static function fnTanh($iX) 
  {
    // tslint:disable-next-line:no-any
    if (function_exists('tanh')) {
      // tslint:disable-next-line:no-any
      return tanh($iX);
    }
    if ($iX === INF) {
      return 1;
    } else if ($iX === -INF) {
      return -1;
    } else {
      $fE2x = exp(2 * $iX);
      return ($fE2x - 1) / ($fE2x + 1);
    }
  }
  
  public static function fnSizeToSquarishShape($iSize)
  {
    for ($iA = floor(sqrt($iSize)); $iA > 1; --$iA) {
      if ($iSize % $iA === 0) {
        return [$iA, $iSize / $iA];
      }
    }
    return [1, $iSize];
  }
  
  public static function fnCreateShuffledIndices($iN) 
  {
    $iShuffledIndices = array_fill(0, $iN, 0);
    for ($iI = 0; $iI < $iN; ++$iI) {
      $iShuffledIndices[$iI] = $iI;
    }
    self::fnShuffle($iShuffledIndices);
    return $iShuffledIndices;
  }
  
  public static function fnRightPad($sA, $iSize) 
  {
    if ($iSize <= strlen($sA)) {
      return $sA;
    }
    return $sA + str_repeat(' ', $iSize - strlen($sA));
  }
  
  /*
  public static function fnRepeatedTry($fnCheckFn, $fnDelayFn = (counter: number) => 0, $iMaxCounter)
  {
    return new Promise<void>((resolve, reject) => {
      let tryCount = 0;

      const tryFn = () => {
        if (checkFn()) {
          resolve();
          return;
        }

        tryCount++;

        const nextBackoff = delayFn(tryCount);

        if (maxCounter != null && tryCount >= maxCounter) {
          reject();
          return;
        }
        setTimeout(tryFn, nextBackoff);
      };

      tryFn();
    });
  }
   */
  
  public static function fnInferFromImplicitShape($aShape, $iSize) 
  {
    $iShapeProd = 1;
    $iImplicitIdx = -1;

    for ($iI = 0; $iI < count($aShape); ++$iI) {
      if ($aShape[$iI] >= 0) {
        $iShapeProd *= $aShape[$iI];
      } else if ($aShape[$iI] === -1) {
        if ($iImplicitIdx !== -1) {
          throw new Exception(
            self::fnFormatString(
              "Shapes can only have 1 implicit size. Found -1 at dim ?: and dim ?:",
              $iImplicitIdx,
              $iI
            )
          );
        }
        $iImplicitIdx = $iI;
      } else if ($aShape[$iI] < 0) {
        throw new Exception(
          self::fnFormatString(
            "Shapes can not be < 0. Found ?: at dim ?:",
            $aShape[$iI],
            $iI
          )
        );
      }
    }

    if ($iImplicitIdx === -1) {
      if ($iSize > 0 && $iSize !== $iShapeProd) {
        throw new Exception(
          self::fnFormatString(
            "Size(?:) must match the product of shape ?:",
            $iSize,
            $aShape
          )
        );
      }
      return shape;
    }

    if ($iShapeProd === 0) {
      throw new Exception(
        self::fnFormatString(
          "Cannot infer the missing size in [?:] when there are 0 elements",
          $aShape
        )
      );
    }
    if ($iSize % $iShapeProd !== 0) {
      throw new Exception(
        self::fnFormatString(
          "The implicit shape can't be a fractional number. Got ?: / ?:",
          $iSize,
          $iShapeProd
        )
      );
    }

    $aNewShape = clone $aShape;
    $aNewShape[$iImplicitIdx] = $iSize / $iShapeProd;
    return $aNewShape;
  }
  
  public static function fnSqueezeShape($aShape, $aAxis)
  {
    $aNewShape = [];
    $aKeptDims = [];
    $iJ = 0;
    for ($iI = 0; $iI < count($aShape); ++$iI) {
      if ($aAxis != null) {
        if ($aAxis[$iJ] === $iI && $aShape[$iI] !== 1) {
          throw new Exception(
            self::fnFormatString(
              "Can't squeeze axis ?: since its dim '?:' is not 1",
              $iI,
              $aShape[$iI]
            )
          );
        }
        if (($aAxis[$iJ] == null || $aAxis[$iJ] > $iI) && $aShape[$iI] === 1) {
          array_push($aNewShape, $aShape[$iI]);
          array_push($aKeptDims, $iI);
        }
        if ($aAxis[$iJ] <= $iI) {
          $iJ++;
        }
      }
      if (shape[$iI] !== 1) {
        array_push($aNewShape, $aShape[$iI]);
        array_push($aKeptDims, $iI);
      }
    }
    return ['newShape' => $aNewShape, 'keptDims' => $aKeptDims];
  }
  
  public static function fnGetTypedArrayFromDType($sDtype, $iSize) 
  {
    $aValues = null;
    if ($sDtype == null || $sDtype === 'float32') {
      $aValues = array_fill(0, $iSize, 0);
    } else if ($sDtype === 'int32') {
      $aValues = array_fill(0, $iSize, 0);
    } else if ($sDtype === 'bool') {
      $aValues = array_fill(0, $iSize, 0);
    } else {
      throw new Exception(
        self::fnFormatString(
          "Unknown data type ?:",
          $sDtype
        )
      );
    }
    return $aValues;
  }

  public static function fnCheckComputationForNaN($aVals, $sDtype, $sName)
  {
    if ($sDtype !== 'float32') {
      // Only floating point computations will generate NaN values
      return;
    }
    for ($iI = 0; $iI < count($aVals); $iI++) {
      if (is_nan($aVals[$iI])) {
        throw new Exception(
          self::fnFormatString(
            "The result of the '?:' has NaNs.",
            $sName
          )
        );
      }
    }
  }
  
  public static function fnCheckConversionForNaN($aVals, $sDtype)
  {
    if ($sDtype === 'float32') {
      // NaN is valid for floating point conversions
      return;
    }

    for ($iI = 0; $iI < count($aVals); $iI++) {
      if (is_nan($aVals[$iI])) {
        throw new Exception(
          self::fnFormatString(
            "NaN is not a valid value for dtype: '?:'.",
            $sDtype
          )
        );
      }
    }
  }
  
  public static function fnHasEncodingLoss($sOldType, $sNewType)
  {
    if ($sNewType === 'complex64') {
      return false;
    }
    if ($sNewType === 'float32' && $sOldType !== 'complex64') {
      return false;
    }
    if ($sNewType === 'int32' && $sOldType !== 'float32' && $sOldType !== 'complex64') {
      return false;
    }
    if ($sNewType === 'bool' && $sOldType === 'bool') {
      return false;
    }
    return true;
  }
  
  public static function fnCopyTypedArray($aArray, $sDtype, $bDebugMode)
  {
    if ($sDtype == null || $sDtype === 'float32' || $sDtype === 'complex64') {
      return clone $aArray;
    } else if ($sDtype === 'int32') {
      if ($bDebugMode) {
        self::fnCheckConversionForNaN($aArray, $sDtype);
      }
      return clone $aArray;
    } else if ($sDtype === 'bool') {
      $aBool = array_fill(0, count($aArray), 0);
      for ($iI = 0; $iI < count($aBool); ++$iI) {
        if (round($aArray[$iI]) !== 0) {
          $aBool[$iI] = 1;
        }
      }
      return $aBool;
    } else {
      throw new Exception(
        self::fnFormatString(
          "Unknown data type ?:",
          $sDtype
        )
      );
    }
  }
  
  public static function fnIsTypedArray($mA)
  {
    /*
    return $mA instanceof Float32Array || $mA instanceof Int32Array ||
        $mA instanceof Uint8Array;
     */
    return is_array($mA);
  }
  
  public static function fnBytesPerElement($sDtype)
  {
    if ($sDtype === 'float32' || $sDtype === 'int32') {
      return 4;
    } else if ($sDtype === 'complex64') {
      return 8;
    } else if ($sDtype === 'bool') {
      return 1;
    } else {
      throw new Exception(
        self::fnFormatString(
          "Unknown dtype ?:",
          $sDtype
        )
      );
    }
  }

  public static function fnIsFunction($fnF) {
    /*
    return !!(f && f.constructor && f.call && f.apply);
     */
    return is_callable($fnF);
  }
  
  public static function fnNearestDivisor($iSize, $iStart)
  {
    for ($iI = $iStart; $iI < $iSize; ++$iI) {
      if ($iSize % $iI === 0) {
        return $iI;
      }
    }
    return $iSize;
  }
  
  public static function fnComputeStrides($aShape)
  {
    $iRank = count($aShape);
    if ($iRank < 2) {
      return [];
    }

    // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
    // strides.
    $aStrides = array_fill(0, $iRank - 1, 0);
    $aStrides[$iRank - 2] = $aShape[$iRank - 1];
    for ($iI = $iRank - 3; $iI >= 0; --$iI) {
      $aStrides[$iI] = $aStrides[$iI + 1] * $aShape[$iI + 1];
    }
    return $aStrides;
  }
  
  public static function fnToTypedArray($aA, $sDtype, $bDebugMode)
  {
    if (self::fnNoConversionNeeded($aA, $sDtype)) {
      return $aA;
    }
    if (is_array(a)) {
      $aA = self::fnFlatten($aA);
    }
    return self::fnCopyTypedArray($aA, $sDtype, $bDebugMode);
  }
  
  public static function fnNoConversionNeeded($aA, $sDtype)
  {
    /*
    return (a instanceof Float32Array && dtype === 'float32') ||
        (a instanceof Int32Array && dtype === 'int32') ||
        (a instanceof Uint8Array && dtype === 'bool');
     * 
     */
    return is_array($aA) && (in_array($sDtype, ['float32', 'int32', 'bool']));
  }
  
  public static function fnMakeOnesTypedArray($iSize, $sDtype)
  {
    $aArray = self::fnMakeZerosTypedArray($iSize, $sDtype);
    for ($iI = 0; $iI < count($aArray); $iI++) {
      $aArray[$iI] = 1;
    }
    return $aArray;
  }
  
  public static function fnMakeZerosTypedArray($iSize, $sDtype)
  {
    if ($sDtype == null || $sDtype === 'float32' || $sDtype === 'complex64') {
      return array_fill(0, $iSize, 0);
    } else if ($sDtype === 'int32') {
      return array_fill(0, $iSize, 0);
    } else if ($sDtype === 'bool') {
      return array_fill(0, $iSize, 0);
    } else {
      throw new Exception(
        self::fnFormatString(
          "Unknown data type ?:",
          $sDtype
        )
      );
    }
  }
}
