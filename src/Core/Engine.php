<?php

namespace libTensorFlowPHP\Core;

use libTensorFlowPHP\Core\Profiler;
use libTensorFlowPHP\Core\TapeNode;
use libTensorFlowPHP\Core\ScopeState;
use libTensorFlowPHP\Set;
use Exception;
use Closure;

interface TensorManager {
  function fnRegisterTensor($oTensor);
  function fnRegisterVariable($mVariable);
  function fnDisposeTensor($oTensor);
  function fnMemory();
}

interface ScopeState {
}

class Environment implements TensorManager
{
  public $registeredVariables;
  
  public $refCounter;
  public $nextTapeNodeId = 0;
  public $numBytes = 0;
  public $numTensors = 0;
  public $numDataBuffers = 0;
  public $profiling = false;
  public $activeProfile;
  public $activeTape;
  public $gradientScopeCount = 0;
  public $customGradientDepth = 0;
  public $activeScope;
  public $scopeStack;
  public $keepTensors;
  public $profiler;
  
  function __construct($oBackend, $bSafeMode, $bDebugMode) {
    $this->refCounter = new WeakMap();
    $this->keepTensors = [];//new Set();
    
    $this->activeScope = ['track' => [], 'name' => 'default scope'];
    $this->scopeStack = [$this->activeScope];
    $this->profiler = new Profiler($oBackend);
    $this->activeProfile = [
      'newBytes' => 0, 
      'newTensors' => 0, 
      'peakBytes' => 0, 
      'kernels' => [], 
      'result' => null
    ]; 
  }
  
  public static function fnTidy($mNameOrFn, $fnFn, $bGradMode = false)
  {
    // gradMode Primarily for internal use during backprop
    //          If true, will start a tape if it is the outermost tidy.

    $sName = null;
    
    if ($fnFn == null) {
      // Called with only 1 argument.
      if (is_callable($mNameOrFn)) {
        throw new Exception('Please provide a function to tidy()');
      }
      $fnFn = $mNameOrFn;
    } else {
      // Called with 2 arguments.
      if (is_string($mNameOrFn)) {
        throw new Exception(
            'When calling with two arguments, the first argument ' .
            'to tidy() must be a string');
      }
      if (is_callable($fnFn)) {
        throw new Exception(
            'When calling with two arguments, the 2nd argument ' .
            'to tidy() must be a function');
      }
      $sName = $mNameOrFn;
      // TODO(nsthorat,smilkov): Do operation logging and performance
      // profiling.
    }
    $fnResult;
    
    $fnFunction1 = function() use ($sName, $bGradMode)
    {
      return $this->fnStartScope($sName, $bGradMode);
    };
    
    $fnFunction2 = function() use ($fnResult, $bGradMode, $fnFn)
    {
      $this->fnEndScope($fnResult, $bGradMode);
      return function () use ($fnFn)
      {
        $mResult = $fnFn();
        /*
          if (result instanceof Promise) {
            console.error('Cannot return a Promise inside of tidy.');
          }
         */
        return $mResult;
      };
    };
    
    return $this->fnScopedRun(
      $fnFunction1,
      $fnFunction2
    );
  }
  
  public function fnScopedRun($fnStart, $fnEnd, $fnF)
  {
    $fnStart = Closure::bindTo($fnStart, $this);
    $fnEnd = Closure::bindTo($fnEnd, $this);
    $fnF = Closure::bindTo($fnF, $this);
    
    $fnStart();
    try {
      $mRes = $fnF();
      $fnEnd();
      return $mRes;
    } catch (Exception $oException) {
      $fnEnd();
      throw $oException;
    }
  }
  
  public function fnRunKernel($fnForwardFunc, $mInputs, $fnBackwardsFunc)
  {
    $mResult;
    $aSaved = [];
    $fnSaveFunc = function($mX) use ($aSaved)
    {
      array_push($aSaved, $mX);
      return $mX;
    };
    $sScopeName = $this->activeScope->name;
    $iStartingBytecount = $this->numBytes;
    $iStartingNumTensors = $this->numTensors;

    $fnFunction1 = function()
    {
      return $this->customGradientDepth++;
    };

    $fnFunction2 = function()
    {
      return $this->customGradientDepth--;
    };

    $fnFunction3 = function() use ($fnForwardFunc, &$mResult, $sScopeName, $fnSaveFunc)
    {
      if (!$this->fnDebugMode()) {
        $mResult = $fnForwardFunc($this->backend, $fnSaveFunc);
      } else {
        $fnFunction1 = function() use ($fnForwardFunc)
        {
          $fnForwardFunc($this->backend, $fnSaveFunc);
        };
        $mResult = $this->profiler->fnProfileKernel(
          $sScopeName, 
          $fnFunction1
        );
      }      
    };

    // Stop recording to a tape when running a kernel.
    $this->fnScopedRun(
      $fnFunction1, 
      $fnFunction2,
      $fnFunction3
    );

    if ($this->fnShouldRecord()) {
      $aTapeNode = [
        'id' => $this->nextTapeNodeId++,
        'name' => $sScopeName,
        'inputs' => $mInputs,
        'outputs' => is_array($mResult) ? $mResult : [$mResult]
      ];
      if ($fnBackwardsFunc != null) {
        $aTapeNode['gradient'] = function ($mDy) use ($mDy, $aSaved)
        {
          $fnBackwardsFunc($mDy, $aSaved);
        };
      }
      array_push($this->activeTape, $aTapeNode);
    }

    if ($this->profiling) {
      array_push($this->activeProfile->kernels, [
        'name' => $sScopeName,
        'bytesAdded' => $this->numBytes - $iStartingBytecount,
        'totalBytesSnapshot' => $this->numBytes,
        'tensorsAdded' => $this->numTensors - $iStartingNumTensors,
        'totalTensorsSnapshot' => $this->numTensors,
        'inputShapes' => array_map(function($v) { return $v->shape; }, $mInputs),
        'outputShape' => is_array($mResult) ?
          array_map(function($v) { return $v->shape; }, $mResult) :
          $mResult->shape
      ]);
    }

    return $mResult;
  }
  
  public function fnRegisterTensor($oA) 
  {
    $iRefCount =
      $this->refCounter->fnHas($oA->dataId) ? 
        $this->refCounter->fnGet($oA->dataId) : 
        0;
    $this->numTensors++;
    if ($iRefCount === 0) {
      $this->numDataBuffers++;

      // Don't count bytes for complex numbers as they are counted by their
      // components.
      if ($oA->dtype !== 'complex64') {
        $this->numBytes +=
          Utilities::fnSizeFromShape($oA->shape) *
          Utilities::fnBytesPerElement($oA->dtype);
      }

      $this->backend->fnRegister($oA->dataId, $oA->shape, $oA->dtype);
    }
    $this->refCounter->fnSet($oA->dataId, $iRefCount + 1);
    if (!($oA instanceof Variable)) {
      $this->fnTrack(a);
    }
  }  
  
  public function fnRegisterVariable($mV)
  {
    if ($this->registeredVariables[$mV->name] != null) {
      throw new Error("Variable with name {$mV->name} was already registered");
    }
    $this->registeredVariables[$mV->name] = v;
  }

  public function fnDisposeVariables()
  {
    foreach ($this->registeredVariables as $sVarName => $mV) {
      $this->fnDisposeTensor($mV);
      unset($this->registeredVariables[$sVarName]);
    }
  }

  public function fnMemory()
  {
    $oInfo = $this->backend->fnMemory();
    $oInfo->numTensors = $this->numTensors;
    $oInfo->numDataBuffers = $this->numDataBuffers;
    $oInfo->numBytes = $this->numBytes;
    return $oInfo;
  }

  public function fnProfile($fnQuery)
  {
    $this->profiling = true;

    $iStartBytes = $this->numBytes;
    $iStartNumTensors = $this->numTensors;

    $this->activeProfile->kernels = [];
    $this->activeProfile->result = $fnQuery();

    $this->profiling = false;

    $this->activeProfile->peakBytes = max(...array_map(
      function($oD) { return $oD->totalBytesSnapshot; }, 
      $this->activeProfile->kernels
    ));
    $this->activeProfile->newBytes = $this->numBytes - startBytes;
    $this->activeProfile->newTensors = $this->numTensors - startNumTensors;
    return $this->activeProfile;
  }

  private function fnShouldRecord()
  {
    return $this->activeTape != null && $this->customGradientDepth === 0;
  }

  private function fnAddTapeNode($aInputs, $oResult, $fnGradientsFunc) 
  {
    $aInputsMap = [];
    
    $aInputsMap = array_merge($aInputsMap, $aInputs);

    $fnGradient = function($oDy) use ($fnGradientsFunc)
    {
      $aRes = $fnGradientsFunc($oDy);
      $aResMap = [];
      
      foreach ($aRes as $iIdx => $mR) {
        $aResMap[$iIdx] = function() use ($mR) {
          return $mR;
        };
      }
      
      return $aResMap;
    };

    $oTapeNode = new TapeNode();
    $oTapeNode->id = $this->nextTapeNodeId++;
    $oTapeNode->name = $this->activeScope->name;
    $oTapeNode->inputs = $aInputsMap;
    $oTapeNode->outputs = [$oResult];
    $oTapeNode->gradient = $fnGradient;
    
    array_push($this->activeTape, $oTapeNode);
  }

  public function fnKeep($oResult)
  {
    if (count($this->scopeStack) === 1 && $this->safeMode) {
      throw new Exception(
          'Safe mode is ON. Enclose all tensor operations inside tf.tidy(): ' .
          'tf.tidy(() => {...}) to avoid memory leaks.');
    }
    $this->keepTensors->add($oResult->id);
    return $oResult;
  }

  /**
   * Start a scope. Use this with endScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  public function fnStartScope($sName, $bGradientsMode = false) {
    if ($bGradientsMode && $this->gradientScopeCount === 0) {
      $this->activeTape = [];
    }
    if ($bGradientsMode) {
      $this->gradientScopeCount++;
    }

    $oScopeInfo = new ScopeState();
    $oScopeInfo->track = [];
    $oScopeInfo->name = 'unnamed scope';
    
    if ($sName) {
      $oScopeInfo->name = $sName;
    }
    
    array_push($this->scopeStack, $oScopeInfo);
    $this->activeScope = $oScopeInfo;
  }

  /**
   * End a scope. Use this with startScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  public function fnEndScope($oResult, $bGradientsMode = false) 
  {
    if ($bGradientsMode) {
      $this->gradientScopeCount--;
      if ($this->gradientScopeCount === 0) {
        $this->activeTape = null;
      }
    }

    $aTensorsToKeep = new Set($this->keepTensors);

    $aTensorsToTrackInParent = Utilities::fnGetTensorsInContainer($oResult);
    foreach ($aTensorsToTrackInParent as $oTensor) {
      $aTensorsToKeep->fnAdd($oTensor->id);
    }

    // Dispose the arrays tracked in this scope.
    for ($iI = 0; $iI < count($this->activeScope->track); $iI++) {
      $oTensor = $this->activeScope->track[$iI];
      if ($aTensorsToKeep->fnHas($oTensor->id)) {
        continue;
      }

      if ($this->activeTape != null) {
        array_push($aTensorsToTrackInParent, $oTensor);
      } else {
        $oTensor->fnDispose();
      }
    }

    $oOldScope = array_pop($this->scopeStack);
    if (count($this->scopeStack) === 0) {
      $this->activeScope = new ScopeState();
      $this->activeScope->track = [];
      $this->activeScope->name = 'default scope';
    } else {
      $this->activeScope = $this->scopeStack[count($this->scopeStack) - 1];
    }

    // Track the current result in the parent scope.
     foreach ($aTensorsToTrackInParent as $oTensor) {
      // Only track the tensor if was allocated in the inner scope and is not
      // globally kept.
      if (!$this->keepTensors->fnHas($oTensor->id) &&
          Utilities::fnIsTensorInList($oTensor, $oOldScope->track)) {
        $this->track($oTensor);
      }
    };
  }

  /**
   * Returns gradients of `f` with respect to each of the `xs`. The gradients
   * returned are of the same length as `xs`, but some might be null if `f` was
   * not a function of that `x`. It also takes optional dy to multiply the
   * gradient, which defaults to `1`.
   */
  public function fnGradients($fnF, $aXs, $oDy, $bAllowNoGradients = false)
  {
    Utilities::fnAssert(count($aXs) > 0, 'gradients() received an empty list of xs.');

    return $this->fnTidy(
      'gradients', 
      function() use ($fnF, $aXs, $oDy, $bAllowNoGradients)
      {
        $oY = $fnF();
        Utilities::fnAssert(
          $oY instanceof Tensor,
          'The result y returned by f() must be a tensor.'
        );
        // Filter out the nodes that don't connect x => y.
        $aFilteredTape = Utilities::fnGetFilteredNodesXToY(
          $this->activeTape, 
          $aXs, 
          $oY
        );
        if (!$bAllowNoGradients && count($aFilteredTape) === 0 && count($aXs) > 0) {
          throw new Exception(
              'Cannot compute gradient of y=f(x) with respect to x. Make sure ' .
              'that the f you passed encloses all operations that lead from x ' .
              'to y.');
        }

        $aAccumulatedGradientMap = [];
        $aAccumulatedGradientMap[$oY->id] = ($oDy == null) ? ones($oY->shape) : $oDy;

        // Backprop gradients through the filtered nodes.
        Utilities::fnBackpropagateGradients($aAccumulatedGradientMap, $aFilteredTape);

        $aGrads = array_map(
          function($oX) use ($aAccumulatedGradientMap) 
          { 
            return $aAccumulatedGradientMap[$oX->id];
          },
          $aXs
        );
          
        return [ 'value' => $oY, 'grads' => $aGrads ];
      }, 
      true /* gradientsMode */
    );
  }

  public function fnCustomGrad($fnF)
  {
    Utilities::fnAssert(
      is_callable($fnF), 
      'The f passed in customGrad(f) must be a function.'
    );
    return function (...$aInputs)
    {
      Utilities::fnAssert(
        Utilities::fnArrayEvery(function($oT) { return $oT instanceof Tensor; }, $aInputs),
        'The args passed in customGrad(f)(x1, x2,...) must all be tensors');

      $fnGradientsFunc;
      $oResult;
      
      $this->fnScopedRun(
        function () { $this->customGradientDepth++; }, 
        function () { $this->customGradientDepth--; },
        function () use (&$fnGradientsFunc, &$oResult)
        {
          $bGradientsMode = true;
          $oResult = $this->fnTidy(
            /*f.name*/null, 
            function () 
            {
              $aResult = $fnF(...$aInputs);

              Utilities::fnAssert(
                $aResult['value'] instanceof Tensor,
                'The function f passed in customGrad(f) must return an ' .
                'object where `obj.value` is a tensor');
              Utilities::fnAssert(
                is_callable($aResult['gradFunc']),
                'The function f passed in customGrad(f) must return an ' .
                'object where `obj.gradFunc` is a function.');

              $fnGradientsFunc = $aResult['gradFunc'];

              return $aResult['value'];
            }, 
            $bGradientsMode
          );
        }
      );

      if ($this->fnShouldRecord()) {
        $fnGradFunc = function ($oDy) use ($fnGradientsFunc, $aInputs)
        {
          $mRes = $fnGradientsFunc($oDy);
          $aGrads = is_array($mRes) ? $mRes : [$mRes];
          Utilities::fnAssert(
            count($aGrads) === count($aInputs),
            'The function f passed in customGrad(f) must return an object ' .
            'where `obj.gradFunc` is a function that returns the same ' .
            'number of tensors as inputs passed to f(...).');
          Utilities::fnAssert(
            Utilities::fnArrayEvery(
              function ($oT) { return $oT instanceof Tensor; },
              $aGrads
            ),
            'The function f passed in customGrad(f) must return an object ' .
            'where `obj.gradFunc` is a function that returns a list of ' .
            'only tensors.');
          return $aGrads;
        };
        
        $this->fnAddTapeNode($aInputs, $oResult, $fnGradFunc);
      }
      
      return $oResult;
    };
  }

  // Forwarding to backend.
  public function fnWrite($sDataId, $aValues)
  {
    $this->backend.write($sDataId, $aValues);
  }
  
  public function fnReadSync($sDataId)
  {
    return $this->backend->fnReadSync($sDataId);
  }
  
  public function fnRead($sDataId)
  {
    return $this->backend->fnRead($sDataId);
  }
  
  public function fnFromPixels($oPixels, $iNumChannels)
  {
    return $this->backend->fnFromPixels($oPixels, $iNumChannels);
  }
  
  public function fnTime($fnQuery)
  {
    $fStart = Utilities::fnNow();
    $oTimingInfo = $this->backend->time($fnQuery);
    $oTimingInfo->wallMs = Utilities::fnNow() - $fStart;
    return $oTimingInfo;
  }

  /**
   * Tracks a Tensor in the current scope to be automatically cleaned up
   * when the current scope ends, and returns the value.
   *
   * @param result The Tensor to track in the current scope.
   */
  private function fnTrack($oResult)
  {
    if (count($this->scopeStack) === 1 && $this->safeMode) {
      throw new Exception(
        'Safe mode is ON. Enclose all tensor operations inside tf.tidy(): ' .
        'tf.tidy(() => {op();...}); to avoid memory leaks.');
    }
    array_push($this->activeScope->track, $oResult);
    return $oResult;
  }
}