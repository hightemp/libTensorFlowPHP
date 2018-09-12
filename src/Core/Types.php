<?php

namespace libTensorFlowPHP\Core;

class Types
{
  public static $aDType = [
    'float32' => 'float32',
    'int32' => 'int32',
    'bool' => 'bool'
  ];
  
  public static $aRank = [
    'R0' => 'R0',
    'R1' => 'R1',
    'R2' => 'R2',
    'R3' => 'R3',
    'R4' => 'R4',
    'R5' => 'R5',
    'R6' => 'R6'
  ];
  
  public static $aUpcastInt32AndMap = [
    'float32' => 'float32',
    'int32' => 'int32',
    'bool' => 'int32',
    'complex64' => 'complex64'
  ];

  public static $aUpcastBoolAndMap = [
    'float32' => 'float32',
    'int32' => 'int32',
    'bool' => 'bool',
    'complex64' => 'complex64'
  ];

  public static $aUpcastFloat32AndMap = [
    'float32' => 'float32',
    'int32' => 'float32',
    'bool' => 'float32',
    'complex64' => 'complex64'
  ];

  public static $aUpcastComplex64AndMap = [
    'float32' => 'complex64',
    'int32' => 'complex64',
    'bool' => 'complex64',
    'complex64' => 'complex64'
  ];

  public static $aUpcastTypeMap = [
    'float32' => 'UpcastFloat32AndMap',
    'int32' => 'UpcastInt32AndMap',
    'bool' => 'UpcastBoolAndMap',
    'complex64' => 'UpcastComplex64AndMap'
  ];

  public static function fnUpcastType($sTypeA, $sTypeB)
  {
    return self::$aUpcastTypeMap[$sTypeA][$sTypeB];
  }

  public static function fnSumOutType($sType)
  {
    return self::fnUpcastType($sType, 'int32');
  }
}