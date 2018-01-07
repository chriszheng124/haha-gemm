#ifndef GEMM_LOG_STREAM_BUFFER_H
#define GEMM_LOG_STREAM_BUFFER_H 

#include <android/log.h>
#include <iostream>
#include <streambuf>
#include "include/common.h"

HAHA_GEMM_BEGIN 

class LogStreamBuffer : public std::streambuf {
public:
    LogStreamBuffer(){
        memset(buf_, 0, BUFFER_SIZE);
        setp(buf_, buf_ + BUFFER_SIZE - 1);
    }

    virtual ~LogStreamBuffer(){
        sync();
    }

protected:
    virtual int_type overflow(int_type c){
        if(c != EOF){
            *pptr() = c;
            pbump(1);
        }
        flush_buffer();
        return c;
    }

    virtual int sync(){
        flush_buffer();
        return 0;
    }

private:  
    int flush_buffer(){  
        int len = int(pptr() - pbase());  
        if (len <= 0){
            return 0;
        }  
        buf_[len] = '\0';

#ifdef ANDROID    
        android_LogPriority t = ANDROID_LOG_INFO;  
        __android_log_write(t, "haha_gemm", buf_);  
#else    
        printf("%s", buf_);  
#endif    
  
        pbump(-len);
        return len;
    }  
private:
    static const int BUFFER_SIZE = 4096;
    char buf_[BUFFER_SIZE];
};

HAHA_GEMM_END

#endif
