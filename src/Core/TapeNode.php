<?php

namespace libTensorFlowPHP\Core;

use Exception;

class TapeNode
{
  public $id;
  public $name;
  public $outputs;
  public $inputs;
  // Optional params, defined only for ops with gradient impl.
  public $gradient;
}
