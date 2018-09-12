<?php

namespace libTensorFlowPHP\Core;

use libTensorFlowPHP\Core\Profiler;
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
    if (this.registeredVariables[v.name] != null) {
      throw new Error(`Variable with name ${v.name} was already registered`);
    }
    this.registeredVariables[v.name] = v;
  }

}