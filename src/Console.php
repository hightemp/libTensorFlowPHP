<?php

namespace libTensorFlowPHP;

class Console
{
  public static $aColors = [
    'gray'      => 30,
    'black'     => 30,
    'red'       => 31,
    'green'     => 32,
    'yellow'    => 33,
    'blue'      => 34,
    'magenta'   => 35,
    'cyan'      => 36,
    'white'     => 37,
    'default'   => 39
  ];
  
  public static $aBgColors = [
    'gray'     => 40,
    'black'    => 40,
    'red'      => 41,
    'green'    => 42,
    'yellow'   => 43,
    'blue'     => 44,
    'magenta'  => 45,
    'cyan'     => 46,
    'white'    => 47,
    'default'  => 49,
  ];
  
  public static $aStyles = [
    'default'          => '0',
    'bold'             => 1,
    'faint'            => 2,
    'normal'           => 22,
    'italic'           => 3,
    'notitalic'        => 23,
    'underlined'       => 4,
    'doubleunderlined' => 21,
    'notunderlined'    => 24,
    'blink'            => 5,
    'blinkfast'        => 6,
    'noblink'          => 25,
    'negative'         => 7,
    'positive'         => 27,
  ];
  
  public static $aStylesUnsetFlags = [
    'bold'             => 22,      
    'underlined'       => 24,
    'blink'            => 25,
  ];
  
  public static function warn(...$aArguments)
  {
    echo $aArguments[0];
  }
  
  public static function log($sMessage, $aArguments = [])
  {
    echo self::fnFormatColoredOutput($sMessage, $aArguments);
  }
  
  public static function fnFormatColoredOutput($sString, $aArguments)
  {
    reset($aArguments);
    return preg_replace_callback(
      "/\?:/", 
      function() use ($aArguments)
      {
        $aParams = [];
        $sResult = '';
        
        $aParsedParams = explode(',', key($aArguments));
        $aParams = [];
        foreach ($aParsedParams as $mParam) {
          $mParam = explode(':', $mParam);
          $aParams[$mParam[0]] = $mParam[1];
        }
          
        if (isset($aParams['c']))
          $sResult .= "\x1b[".self::$aColors[$aParams['c']]."m";
        
        if (isset($aParams['b']))
          $sResult .= "\x1b[".self::$aBgColors[$aParams['b']]."m";

        if (isset($aParams['s']))
          $sResult .= "\x1b[".self::$aStyles[$aParams['s']]."m";

        $sResult = json_encode(current($aArguments));
        
        if (isset($aParams['c']))
          $sResult .= "\x1b[39m";
        
        if (isset($aParams['b']))
          $sResult .= "\x1b[49m";

        if (isset($aParams['s']))
          $sResult .= "\x1b[".self::$aStylesUnsetFlags[$aParams['s']]."m";

        next($aArguments);
        
        return $sResult;
      },
      $sString
    );
  }

}
