<?php

namespace libTensorFlowPHP\Core;

use Exception;

class OpHandler
{
  public static $oOpHandler = null;
  
  public static function fnSetOpHandler($oOpHandler) {
    self::$oOpHandler = $oOpHandler;
  }
  
}
