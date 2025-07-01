// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the LVOPENVINO_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// LVOPENVINO_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef LVOPENVINO_EXPORTS
#define LVOPENVINO_API extern "C" __declspec(dllexport)
#else
#define LVOPENVINO_API __declspec(dllimport)
#endif

