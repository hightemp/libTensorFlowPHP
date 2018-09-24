<?php

namespace libTensorFlowPHP\Core;

use libTensorFlowPHP\Core\Tensor;
use libTensorFlowPHP\Core\TapeNode;
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
  
  public static function fnNow()
  {
    /*
    if (typeof performance !== 'undefined') {
      return performance.now();
    } else if (typeof process !== 'undefined') {
      const time = process.hrtime();
      return time[0] * 1000 + time[1] / 1000000;
    } else {
      throw new Error(
          'Cannot measure time in this environment. You should run tf.js ' +
          'in the browser or in Node.js');
    }
     * 
     */
    return microtime(true);
  }
  
  public static function fnAssertTypesMatch($oA, $oB)
  {
    self::fnAssert(
      $oA->dtype === $oB->dtype, 
      "The dtypes of the first({$oA->dtype}) and" .
      " second({$oB->dtype}) input must match"
    );
  }

  public static function fnIsTensorInList($oTensor, $aTensorList)
  {
    for ($iI = 0; $iI < count($aTensorList); $iI++) {
      if ($aTensorList[$iI]->id === $oTensor->id) {
        return true;
      }
    }
    return false;
  }

  public static function fnFlattenNameArrayMap($aNameArrayMap, $aKeys)
  {
    $aXs = [];
    if ($aNameArrayMap instanceof Tensor) {
      array_push($aXs, $aNameArrayMap);
    } else {
      $aXMap = $aNameArrayMap;
      for ($iI = 0; $iI < count($aKeys); $iI++) {
        array_push($aXs, $aXMap[$aKeys[$iI]]);
      }
    }
    return $aXs;
  }

  public static function fnUnflattenToNameArrayMap($aKeys, $aFlatArrays)
  {
    if (count($aKeys) !== count($aFlatArrays)) {
      throw new Exception(
        "Cannot unflatten Tensor[], keys and arrays are not of same length.");
    }
    $aResult = [];
    for ($iI = 0; $iI < count($aKeys); $iI++) {
      $aResult[$aKeys[$iI]] = $aFlatArrays[$iI];
    }
    return $aResult;
  }

  /**
   * Extracts any `Tensor`s found within the provided object.
   *
   * @param container an object that may be a `Tensor` or may directly contain
   *   `Tensor`s, such as a `Tensor[]` or `{key: Tensor, ...}`. In general it
   *   is safe to pass any object here, except that `Promise`s are not
   *   supported.
   * @returns An array of `Tensors` found within the passed object. If the
   *   argument is simply a `Tensor', a list containing that `Tensor` is
   *   returned. If the object is not a `Tensor` or does not
   *   contain `Tensors`, an empty list is returned.
   */
  public static function fnGetTensorsInContainer($oResult)
  {
    $aList = [];
    $oSeen = new Set();
    self::fnWalkTensorContainer($oResult, $aList, $oSeen);
    return $aList;
  }

  public static function fnWalkTensorContainer($oContainer, $aList, $oSeen)
  {
    if ($oContainer == null) {
      return;
    }
    if ($oContainer instanceof Tensor) {
      array_push($aList, $oContainer);
      return;
    }
    if (!self::fnIsIterable($oContainer)) {
      return;
    }
    // Iteration over keys works also for arrays.
    $oIterable = $oContainer;
    foreach ($oIterable as $sK => $sVal) {
      if (!$oSeen->fnHas($sVal)) {
        $oSeen->fnAdd($sVal);
        self::fnWalkTensorContainer($sVal, $aList, $oSeen);
      }
    }
  }

  // tslint:disable-next-line:no-any
  public static function fnIsIterable($oObj)
  {
    return is_array($oObj) || is_object($oObj);
  }

  public static function fnGetFilteredNodesXToY($aTape, $aXs, $oY)
  {
    // Forward pass to compute all the nodes and Tensors that are transitively a
    // function of x.
    $aTensorsFromX = [];
    $aNodesFromX = [];
    
    for ($iI = 0; $iI < count($aXs); $iI++) {
      $aTensorsFromX[$aXs[$iI]->id] = true;
    }

    for ($iI = 0; $iI < count($aTape); $iI++) {
      $oNode = $aTape[$iI];
      $aNodeInputs = $oNode->inputs;
      foreach ($aNodeInputs as $sInputName => $oInput) {
        $bAnyInputFromX = false;
        for ($iJ = 0; $iJ < count($aXs); $iJ++) {
          if ($aTensorsFromX[$oInput->id]) {
            foreach ($oNode->outputs as $oOutput) {
              $aTensorsFromX[$oOutput->id] = true;
            }
            $bAnyInputFromX = true;
            $aNodesFromX[$oNode->id] = true;
            break;
          }
        }

        if ($bAnyInputFromX) {
          break;
        }
      }
    }

    // Backward pass to find all of the nodes and Tensors that lead to y.
    $aTensorsLeadToY = [];
    $aTensorsLeadToY[$oY->id] = true;
    $aNodesToY = [];

    for ($iI = count($aTape) - 1; $iI >= 0; $iI--) {
      $oNode = $aTape[$iI];
      $aNodeInputs = $oNode->inputs;

      // If any of the outputs lead to y, mark all of the inputs as leading to y.
      for ($iJ = 0; $iJ < count($oNode->outputs); $iJ++) {
        if ($aTensorsLeadToY[$oNode->outputs[$iJ]->id]) {
          foreach ($aNodeInputs as $sInputName => $oV) {
            $aTensorsLeadToY[$oV->id] = true;
            $aNodesToY[$oNode->id] = true;
          }
          break;
        }
      }
    }

    // Return the paths that come from x and lead to y.
    $aFilteredTape = [];
    for ($iI = 0; $iI < count($aTape); $iI++) {
      $oNode = $aTape[$iI];

      if ($aNodesFromX[$oNode->id] && $aNodesToY[$oNode->id]) {
        // Prune the inputs from the node that aren't a function of x.
        $aPrunedInputs = [];
        foreach ($oNode->inputs as $sInputName => $oNodeInput) {
          if ($aTensorsFromX[$oNodeInput->id]) {
            $aPrunedInputs[$sInputName] = $oNodeInput;
          }
        }

        // Copy the node and overwrite inputsAndArgs to the pruned version.
        $oPrunedNode = new TapeNode();
        $oPrunedNode->inputs = $aPrunedInputs;
        $oPrunedNode->outputs = $oNode->outputs;

        array_push($aFilteredTape, $oPrunedNode);
      }
    }

    return $aFilteredTape;
  }

  /**
   * Backpropagate gradients through the filtered TapeNodes.
   *
   * @param tensorAccumulatedGradientMap A map of Tensor to its gradient. This map
   * is mutated by this method.
   * @param filteredTape The filtered TapeNodes to backprop through.
   */
  public static function fnBackpropagateGradients($aTensorAccumulatedGradientMap, $aFilteredTape)
  {
    // Walk the tape backward and keep a map of Tensor to its gradient.
    for ($iI = count($aFilteredTape) - 1; $iI >= 0; $iI--) {
      $oNode = $aFilteredTape[$iI];

      $aDys = [];
      foreach ($oNode->outputs as $oO) {
        $oGradTensor = $aTensorAccumulatedGradientMap[$oO->id];
        if ($oGradTensor != null) {
          array_push($aDys, $oGradTensor);
        } else {
          // This particular output is not in the back-propagation subgraph, so it
          // does not affect the final output, thus we put zeros for its dy.
          $oDy = Tensor::fnMake(
              $oO->shape, 
              [ 'values' => self::fnMakeZerosTypedArray($oO->size, $oO->dtype)],
              $oO->dtype);
          array_push($aDys, $oDy);
        }
      };

      if ($oNode->gradient == null) {
        throw new Exception(
            "Cannot compute gradient: gradient function not found " .
            "for {$oNode->name}.");
      }

      // Backprop dy through this node and accumulate gradients over the inputs.
      $aInputGradients =
          // Grad functions of ops with single outputs expect a dy, while ops
          // with multiple outputs expect dys (array of dy).
          $oNode->gradient(count($oNode->outputs) === 1 ? $aDys[0] : $aDys);
      foreach ($oNode->inputs as $sInputName => $oInput) {
        if (!isset($aInputGradients[$sInputName])) {
          throw new Exception(fnFormatString(
            "Cannot backprop through input $sInputName. " .
            "Available gradients found: ?:.", 
            array_keys($aInputGradients)));
        }

        // Call the gradient function.
        $oDx = $aInputGradients[$sInputName]();
        $oX = $oNode->inputs[$sInputName];
        if (!self::fnArraysEqual($oDx->shape, $oX->shape)) {
          throw new Exception(
              "Error in gradient for op {$oNode->name}. The gradient of input " .
              "'$sInputName' has shape '{$oDx->shape}', which does not match " .
              "the shape of the input '{$oX->shape}'");
        }

        if ($aTensorAccumulatedGradientMap[$oX->id] == null) {
          $aTensorAccumulatedGradientMap[$oX->id] = $oDx;
        } else {
          $oCurGradient = $aTensorAccumulatedGradientMap[$oX->id];
          $aTensorAccumulatedGradientMap[$oX->id] = $oCurGradient->fnAdd($oDx);
          $oCurGradient->fnDispose();
        }
      }
    }
  }
  
  public static function fnArrayEvery(callable $callback, array $arr) 
  {
    foreach ($arr as $element) {
      if (!$callback($element)) {
        return FALSE;
      }
    }
    return TRUE;
  }
}