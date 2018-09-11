<?php

namespace libTensorFlowPHP\Core;

use Exception;
use libTensorFlowPHP\Console;
use libTensorFlowPHP\Core\Utilities;

class Profiler
{
  public $logger;
          
  function __construct($oBackendTimer, $oLogger) 
  {
    if ($oLogger == null) {
      $this->logger = new Logger();
    }
  }

  public function fnProfileKernel($sName, $fnF)
  {
    $mResult;
    $fnHoldResultWrapperFn = function() use (&$mResult, $fnF) {
      $mResult = $fnF();
    };
    $oTimer = $this->backendTimer->fnTime($fnHoldResultWrapperFn);

    $aResults = is_array($mResult) ? $mResult : [$mResult];
    foreach ($aResults as $oR) {
      $aVals = $oR->fnDataSync();
      Utilities::fnCheckComputationForNaN($aVals, $oR->dtype, $sName);

      $oTimer->fnThen(
        function ($oTiming) use ($sName, $oR, $aVals) 
        {
          $this->logger->fnLogKernelProfile($sName, $oR, $aVals, $oTiming->kernelMs);
        }
      );
      
    }

    return $mResult;
  }
}

class Logger 
{
  public function fnLogKernelProfile($sName, $oResult, $aVals, $iTimeMs) 
  {
    $sTime = Utilities::fnRightPad("{$iTimeMs}ms", 9);
    $sPaddedName = Utilities::fnRightPad($sName, 25);
    $iRank = $oResult->rank;
    $iSize = $oResult->size;
    $sShape = Utilities::fnRightPad(json_encode($oResult->shape), 14);
    Console::log(
      "?:\t?:\t?:\t?:",
      [
        's:bold' => $sPaddedName, 
        'c:red' => $sTime, 
        'c:blue' => "$iRank:D $sShape", 
        'c:orange' => $iSize
      ]
    );
  }
}

