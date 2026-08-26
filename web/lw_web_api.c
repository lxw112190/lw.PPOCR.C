#include "lw_infer.h"
#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define LW_WEB_API EMSCRIPTEN_KEEPALIVE
#else
#define LW_WEB_API
#endif
#include <stddef.h>
#include <stdlib.h>
static lw_ocr* g_ocr;
static lw_error g_error;
LW_WEB_API int lw_web_init(int use_classifier){lw_ocr_options o;lw_ocr_free(g_ocr);g_ocr=NULL;lw_error_init(&g_error);lw_ocr_options_init(&o);o.use_direction_classification=use_classifier?1u:0u;return (int)lw_ocr_create("/models/det.lwm",use_classifier?"/models/cls.lwm":NULL,"/models/rec.lwm","/models/ppocr_keys.txt",&o,&g_ocr,&g_error);}
LW_WEB_API void lw_web_shutdown(void){lw_ocr_free(g_ocr);g_ocr=NULL;}
LW_WEB_API int lw_web_get_info(lw_ocr_info* i){if(!i)return (int)LW_STATUS_INVALID_ARGUMENT;lw_ocr_info_init(i);return (int)lw_ocr_get_info(g_ocr,i);}
LW_WEB_API int lw_web_run(const uint8_t* s,uint32_t n,uint32_t w,uint32_t h,uint32_t stride,lw_ocr_line* l,uint32_t lc,char* t,uint32_t tc,lw_ocr_result* r){lw_error_init(&g_error);if(r)lw_ocr_result_init(r);return (int)lw_ocr_run_bgr_u8(g_ocr,s,n,w,h,stride,l,lc,t,tc,r,&g_error);}
LW_WEB_API const lw_error* lw_web_last_error(void){return &g_error;}
LW_WEB_API void* lw_web_malloc(size_t n){return malloc(n);}
LW_WEB_API void lw_web_free(void* p){free(p);}
