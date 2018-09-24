<?php

namespace libTensorFlowPHP;

use Exception;

class Set
{
  public $aArray = [];
  
  public function __construct($aArray=[]) 
  {
    $this->fnAdd($aArray);
  }
  
  public function fnAdd($mValue)
  {
    if (is_array($mValue)) {
      foreach ($mValue as $sItem) {
        $this->fnAdd($sItem);
      }
    } else {
      array_push($this->aArray, $mValue);
      $this->aArray = array_unique($this->aArray);
    }
  }
}
