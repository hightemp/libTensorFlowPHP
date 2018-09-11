<?php

namespace libTensorFlowPHP\Core;

use libTensorFlowPHP\Core\Profiler;
use Exception;

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
  
  tidy<T extends TensorContainer>(
      nameOrFn: string|ScopeFn<T>, fn?: ScopeFn<T>, gradMode = false): T {
    // gradMode Primarily for internal use during backprop
    //          If true, will start a tape if it is the outermost tidy.

    let name: string = null;
    if (fn == null) {
      // Called with only 1 argument.
      if (typeof nameOrFn !== 'function') {
        throw new Error('Please provide a function to tidy()');
      }
      fn = nameOrFn;
    } else {
      // Called with 2 arguments.
      if (typeof nameOrFn !== 'string' && !(nameOrFn instanceof String)) {
        throw new Error(
            'When calling with two arguments, the first argument ' +
            'to tidy() must be a string');
      }
      if (typeof fn !== 'function') {
        throw new Error(
            'When calling with two arguments, the 2nd argument ' +
            'to tidy() must be a function');
      }
      name = nameOrFn as string;
      // TODO(nsthorat,smilkov): Do operation logging and performance
      // profiling.
    }
    let result: T;
    return this.scopedRun(
        () => this.startScope(name, gradMode),
        () => this.endScope(result, gradMode), () => {
          result = fn();
          if (result instanceof Promise) {
            console.error('Cannot return a Promise inside of tidy.');
          }
          return result;
        });
  }
}