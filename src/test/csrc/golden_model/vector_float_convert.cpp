#include "../include/gm_common.h"
#include "../include/vfpu_functions.h"
#include <typeinfo>
#include <stdint.h>

static uint32_t bf16_to_f32_bits(uint16_t src) {
  uint16_t exp = (src >> 7) & 0xff;
  uint16_t frac = src & 0x7f;
  if (exp == 0xff && frac != 0) {
    if ((frac & 0x40) == 0) {
      softfloat_exceptionFlags |= softfloat_flag_invalid;
    }
    return 0x7fc00000;
  }
  return ((uint32_t)src) << 16;
}

static bool bf16_round_up(uint16_t kept, bool round, bool sticky, bool sign) {
  bool guard = kept & 0x1;
  bool inexact = round || sticky;
  switch (softfloat_roundingMode) {
    case softfloat_round_near_even:
      return round && (sticky || guard);
    case softfloat_round_minMag:
      return false;
    case softfloat_round_min:
      return inexact && sign;
    case softfloat_round_max:
      return inexact && !sign;
    case softfloat_round_near_maxMag:
      return round;
    case softfloat_round_odd:
      return inexact && !guard;
    default:
      return false;
  }
}

static uint16_t f32_to_bf16_bits(uint32_t src) {
  bool sign = (src >> 31) & 0x1;
  uint32_t exp = (src >> 23) & 0xff;
  uint32_t frac = src & 0x7fffff;
  if (exp == 0xff) {
    if (frac != 0) {
      if ((frac & 0x400000) == 0) {
        softfloat_exceptionFlags |= softfloat_flag_invalid;
      }
      return 0x7fc0;
    }
    return (uint16_t)(src >> 16);
  }

  uint16_t kept = (uint16_t)(src >> 16);
  bool round = (src >> 15) & 0x1;
  bool sticky = (src & 0x7fff) != 0;
  bool inexact = round || sticky;
  uint16_t rounded = kept + (bf16_round_up(kept, round, sticky, sign) ? 1 : 0);
  bool overflow = ((rounded >> 7) & 0xff) == 0xff;
  bool underflow = exp == 0 && (((rounded >> 7) & 0xff) == 0) && inexact;

  if (overflow) {
    softfloat_exceptionFlags |= softfloat_flag_overflow | softfloat_flag_inexact;
    bool roundToMin =
      softfloat_roundingMode == softfloat_round_minMag ||
      (softfloat_roundingMode == softfloat_round_max && sign) ||
      (softfloat_roundingMode == softfloat_round_min && !sign) ||
      softfloat_roundingMode == softfloat_round_odd;
    return sign ? (roundToMin ? 0xff7f : 0xff80) : (roundToMin ? 0x7f7f : 0x7f80);
  }

  if (underflow) {
    softfloat_exceptionFlags |= softfloat_flag_underflow;
  }
  if (inexact) {
    softfloat_exceptionFlags |= softfloat_flag_inexact;
  }
  return rounded;
}


//                               width of output
ElementOutput VGMFloatCvt::calculation_e8(ElementInput input) {
  fp_set_rm(input.rm);
  fp_clear_exception();
  ElementOutput output;
  switch(input.fuOpType) {
    // widen 8->16
    case VFWCVT_FXUV: //ui8 -> f16
     output.result = ui32_to_f16((uint32_t)input.src1).v;  break;
    case VFWCVT_FXV:  //i8 -> f16 
      output.result = i32_to_f16((int32_t)(int8_t)input.src1).v;  break; //todo
    default:
      printf("VFConvert Unsupported fuOpType %d\n", input.fuOpType);
      exit(1);
  }
  
  output.fflags = softfloat_exceptionFlags & 0x1f;
  if (verbose) { display_calculation(typeid(this).name(), __func__, input, output); }
  return output;
}

ElementOutput VGMFloatCvt::calculation_e16(ElementInput input) {
  fp_set_rm(input.rm);
  fp_clear_exception();
  ElementOutput output;
  switch(input.fuOpType) {
    // single 16->16
    case VFCVT_XUFV: // f16->ui16
      output.result = f16_to_ui16(i2f16((uint16_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFCVT_XFV:  // f16->si16      
      output.result = f16_to_i16(i2f16((uint16_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFCVT_FXUV: //ui16-> f16
      output.result = ui32_to_f16((uint32_t)input.src1).v;  break;
    case VFCVT_FXV: //i16 -> f16
      output.result = i32_to_f16((int32_t)(int16_t)input.src1).v;  break;
    case VFCVT_RTZ_XUFV: // f16->Ui16 trun
      output.result = f16_to_ui16(i2f16((uint16_t)input.src1), softfloat_round_minMag, true);  break; 
    case VFCVT_RTZ_XFV:  // f16->Ui16 trun
      output.result = f16_to_i16(i2f16((uint16_t)input.src1), softfloat_round_minMag, true);  break;

    case VFRSQRT7: // f16->f16
      output.result = f16_rsqrte7(i2f16((uint16_t)input.src1)).v; break;
    case VFREC7:  // f16->f16
      output.result = f16_recip7(i2f16((uint16_t)input.src1)).v; break;

    // widen 16->32
    case VFWCVT_XUFV: //f16->ui32 
      output.result = f16_to_ui32(i2f16((uint16_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFWCVT_XFV:  //f16->i32  
      output.result = f16_to_i32(i2f16((uint16_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFWCVT_FXUV: //ui16 -> f32
       output.result = ui32_to_f32((uint32_t)input.src1).v;  break;
    case VFWCVT_FXV:  //i16 -> f32 
      output.result = i32_to_f32((int32_t)(int16_t)input.src1).v;  break;
    case VFWCVT_FFV:  //f16 -> f32 
      output.result = f16_to_f32(i2f16((uint16_t)input.src1)).v;
      break;
    case VFWCVTBF16_FFV: //bf16 -> f32
      output.result = bf16_to_f32_bits((uint16_t)input.src1);
      break;
    case VFWCVT_RTZ_XUFV: //f16 -> ui32 trun 
      output.result = f16_to_ui32(i2f16((uint16_t)input.src1), softfloat_round_minMag, true);  break;
    case VFWCVT_RTZ_XFV: //f16 -> i32 trun  
      output.result = f16_to_i32(i2f16((uint16_t)input.src1), softfloat_round_minMag, true);  break;
    // norrow 16->8
    case VFNCVT_XUFW: // f16 ->ui8
      output.result = f16_to_ui8(i2f16((uint16_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFNCVT_XFW: // f16 ->i8
      output.result = f16_to_i8(i2f16((uint16_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFNCVT_RTZ_XUFW: //f16 -> ui8 trun
      output.result = f16_to_ui8(i2f16((uint16_t)input.src1), softfloat_round_minMag, true);  break; 
    case VFNCVT_RTZ_XFW:  //f16 -> i8 trun
      output.result = f16_to_i8(i2f16((uint16_t)input.src1), softfloat_round_minMag, true);  break; 
    default:
      printf("VFConvert Unsupported fuOpType %d\n", input.fuOpType);
      exit(1);
  }
  
  output.fflags = softfloat_exceptionFlags & 0x1f;
  if (verbose) { display_calculation(typeid(this).name(), __func__, input, output); }
  return output;
}

ElementOutput VGMFloatCvt::calculation_e32(ElementInput input) {
  fp_set_rm(input.rm);
  fp_clear_exception();
  ElementOutput output;
  switch(input.fuOpType) {
    //single 32->32
    case VFCVT_XUFV: // f32->ui32
      output.result = f32_to_ui32(i2f32((uint32_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFCVT_XFV:  // f32->i32      
      output.result = f32_to_i32(i2f32((uint32_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFCVT_FXUV: //ui32-> f32
      output.result = ui32_to_f32((uint32_t)input.src1).v;  break;
    case VFCVT_FXV: //i32 -> f32
      output.result = i32_to_f32((uint32_t)input.src1).v;  break;
    case VFCVT_RTZ_XUFV: // f32->Ui32 trun
      output.result = f32_to_ui32(i2f32((uint32_t)input.src1), softfloat_round_minMag, true);  break; 
    case VFCVT_RTZ_XFV:  // f32->Ui32 trun
      output.result = f32_to_i32(i2f32((uint32_t)input.src1), softfloat_round_minMag, true);  break; 

    case VFRSQRT7: // f32->f32
      output.result = f32_rsqrte7(i2f32((uint32_t)input.src1)).v; break;
    case VFREC7:  // f32->f32
      output.result = f32_recip7(i2f32((uint32_t)input.src1)).v; break;

    // widen 32 -> 64
    case VFWCVT_XUFV: //f32->ui64 
      output.result = f32_to_ui64(i2f32((uint32_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFWCVT_XFV:  //f32->i64  
      output.result = f32_to_i64(i2f32((uint32_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFWCVT_FXUV: //ui32 -> f64
      output.result = ui32_to_f64((uint32_t)input.src1).v;  break; 
    case VFWCVT_FXV:  //i32 -> f64 
      output.result = i32_to_f64((uint32_t)input.src1).v;  break; 
    case VFWCVT_FFV:  //f32 -> f64 
      output.result = f32_to_f64(i2f32((uint32_t)input.src1)).v;   
      break; 

    case VFWCVT_RTZ_XUFV: //f32 -> ui64 trun 
      output.result = f32_to_ui64(i2f32((uint32_t)input.src1), softfloat_round_minMag, true);  break;
    case VFWCVT_RTZ_XFV: //f32 -> i64 trun  
      output.result = f32_to_i64(i2f32((uint32_t)input.src1), softfloat_round_minMag, true);  break;
    // norrow 32 -> 16
    case VFNCVT_XUFW: // f32 ->ui16
      output.result = f32_to_ui16(i2f32((uint32_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFNCVT_XFW: // f32 ->i16
      output.result = f32_to_i16(i2f32((uint32_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFNCVT_FXUW: // ui32 ->f16
     output.result = ui32_to_f16((uint32_t)input.src1).v;  break;
    case VFNCVT_FXW:  // i32 ->f16
      output.result = i32_to_f16((uint32_t)input.src1).v;  break;
    case VFNCVT_FFW:  // f32 ->f16
      output.result = f32_to_f16(i2f32((uint32_t)input.src1)).v;  break;
    case VFNCVTBF16_FFW: // f32 ->bf16
      output.result = f32_to_bf16_bits((uint32_t)input.src1);  break;
    case VFNCVT_ROD_FFW:// f32 ->f16 rounding towards odd ？？？
      softfloat_roundingMode = softfloat_round_odd;
      output.result = f32_to_f16(i2f32((uint32_t)input.src1)).v;  
      break; 
    case VFNCVT_RTZ_XUFW: //f32 -> ui16 trun
      output.result = f32_to_ui16(i2f32((uint32_t)input.src1), softfloat_round_minMag, true);  break; 
    case VFNCVT_RTZ_XFW:  //f32 -> i16 trun
      output.result = f32_to_i16(i2f32((uint32_t)input.src1), softfloat_round_minMag, true);  break; 
    default:
      printf("VFConvert Unsupported fuOpType %d\n", input.fuOpType);
      exit(1);
  }
  
  output.fflags = softfloat_exceptionFlags & 0x1f;
  if (verbose) { display_calculation(typeid(this).name(), __func__, input, output); }
  return output;
}

ElementOutput VGMFloatCvt::calculation_e64(ElementInput input) {
  fp_set_rm(input.rm);
  fp_clear_exception();
  ElementOutput output;
  switch(input.fuOpType) {
    // single 64 -> 64
    case VFCVT_XUFV: // f64->ui64
      output.result = f64_to_ui64(i2f64((uint64_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFCVT_XFV:  // f64->si64      
      output.result = f64_to_i64(i2f64((uint64_t)input.src1), softfloat_roundingMode, true);  break; 
    case VFCVT_FXUV: //ui64-> f64
      output.result = ui64_to_f64((uint64_t)input.src1).v;  break;
    case VFCVT_FXV: //i64 -> f64
      output.result = i64_to_f64((uint64_t)input.src1).v;  
      break;
    case VFCVT_RTZ_XUFV: // f64->Ui64 trun
      output.result = f64_to_ui64(i2f64((uint64_t)input.src1), softfloat_round_minMag, true);  break; 
    case VFCVT_RTZ_XFV:  // f64->Ui64 trun
      output.result = f64_to_i64(i2f64((uint64_t)input.src1), softfloat_round_minMag, true);
      break;


    case VFRSQRT7: // f64->f64
      output.result = f64_rsqrte7(i2f64((uint64_t)input.src1)).v;
      break;
    case VFREC7:  // f64->f64
      output.result = f64_recip7(i2f64((uint64_t)input.src1)).v; 
      break;
    
    // norrow 64->32
    case VFNCVT_XUFW: // f64 ->ui32
      output.result = f64_to_ui32(i2f64((uint64_t)input.src1), softfloat_roundingMode, true);  break;
    case VFNCVT_XFW: // f64 ->i32
      output.result = f64_to_i32(i2f64((uint64_t)input.src1), softfloat_roundingMode, true);  break;
    case VFNCVT_FXUW: // ui64 ->f32
      output.result = ui64_to_f32((uint64_t)input.src1).v;  break;
    case VFNCVT_FXW:  // i64 ->f32
      output.result = i64_to_f32((uint64_t)input.src1).v;  break;
    case VFNCVT_FFW:  // f64 ->f32
      output.result = f64_to_f32(i2f64((uint64_t)input.src1)).v;  break;
    case VFNCVT_ROD_FFW:// f64 ->f32 rounding towards odd ???
      softfloat_roundingMode = softfloat_round_odd;
      output.result = f64_to_f32(i2f64((uint64_t)input.src1)).v;  
      break; 
    case VFNCVT_RTZ_XUFW: //f64 -> ui32 trun
      output.result = f64_to_ui32(i2f64((uint64_t)input.src1), softfloat_round_minMag, true);  break;
    case VFNCVT_RTZ_XFW:  //f64 -> i32 trun
      output.result = f64_to_i32(i2f64((uint64_t)input.src1), softfloat_round_minMag, true);  break;
    default:
      printf("VFConvert Unsupported fuOpType %d\n", input.fuOpType);
      exit(1);
  }
  
  output.fflags = softfloat_exceptionFlags & 0x1f;
  if (verbose) { display_calculation(typeid(this).name(), __func__, input, output); }
  return output;
}
