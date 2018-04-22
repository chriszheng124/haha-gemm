#include <linux/time.h>
#include <sys/time.h>
#include "PerfUtil.h"

long PerfUtil::GetCurrentTimeMs() {
    timeval currTime;
    gettimeofday(&currTime, NULL);
    return currTime.tv_sec * 1000 + currTime.tv_usec/1000;
}
