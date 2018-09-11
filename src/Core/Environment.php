<?php

namespace libTensorFlowPHP\Core;

use Exception;
use libTensorFlowPHP\Console;

class Environment
{
  public $features;
  public $engines;
  public $globalEngine;
  public $registry;
  
  public $backendName;
  
  function __construct($aFeatures = []) {
    if ($aFeatures != null) {
      $this->features = $aFeatures;
    }

    if ($this->get('DEBUG')) {
      Console::warn(
        'Debugging mode is ON. The output of every math call will ' +
        'be downloaded to CPU and checked for NaNs. ' +
        'This significantly impacts performance.'
      );
    }
    
  }
}

