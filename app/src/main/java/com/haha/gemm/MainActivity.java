package com.haha.gemm;

import android.app.Activity;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class MainActivity extends Activity implements View.OnClickListener{
    private HandlerThread mHandlerThread = new HandlerThread("test-gemm");
    private Handler mHandler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        findViewById(R.id.id_test_gemm).setOnClickListener(this);
        findViewById(R.id.id_test_gpu).setOnClickListener(this);

        mHandlerThread.start();
        mHandler = new Handler(mHandlerThread.getLooper());

        TextView tv = (TextView) findViewById(R.id.sample_text);
        tv.setText(stringFromJNI());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mHandlerThread.quitSafely();
    }

    @Override
    public void onClick(final View v) {
        if(v.getId() == R.id.id_test_gemm){
            mHandler.post(new Runnable() {
                @Override
                public void run() {
                    testGemm();
                }
            });
        }else if(v.getId() == R.id.id_test_gpu){
            mHandler.post(new Runnable() {
                @Override
                public void run() {
                    testG(v);
                }
            });
        }
    }

    private void testG(View view)
    {
        AssetManager am = getAssets();
        try {
//            InputStream is = am.open("vector_sum_kernel.cl");
//            InputStream is = am.open("vector_sum_kernel_int.cl");
//            InputStream is = am.open("mat_transpose_kernel.cl");

//            InputStream is = am.open("gemm_naive.cl");
//            InputStream is = am.open("gemm_tiling.cl");
//            InputStream is = am.open("gemm_tiling_vec_image_4x4.cl");
//            InputStream is = am.open("gemm_tiling_vec_image_4x8.cl");

            InputStream is = am.open("gemm_tiling_vec_image_8x8.cl");
//            InputStream is = am.open("gemm_tiling_vec_image_a_packed_8x8.cl");

//            InputStream is = am.open("gemm_tiling_vec_a_as_image_8x8.cl");
//            InputStream is = am.open("gemm_tiling_vec_image_4x8_prefetch.cl");
//            InputStream is = am.open("gemm_tiling_vec_all_image_4x8.cl");
//            InputStream is = am.open("gemm_tiling_vec_image_int_4x8.cl");
//            InputStream is = am.open("gemm_tiling_vec_4x8.cl");
//            InputStream is = am.open("gemm_tiling_vec_image_unrolling_4x8.cl");

            String kernelCode = convertInputStreamToString(is);

            sgemm(kernelCode);

//            testMatTranspose(kernelCode);
//            testVectorSum(kernelCode);
        } catch(IOException e) {
            Log.d("oclDebug",e.toString());
        }
    }

    private String convertInputStreamToString(InputStream in) {
        BufferedReader br;
        StringBuffer outString = new StringBuffer();
        br = new BufferedReader(new InputStreamReader(in));
        try{
            String read = br.readLine();

            while(read != null) {
                outString.append(read);
                read = br.readLine();
            }
        } catch(IOException e) {
            Log.d("oclDebug",e.toString());
        }

        return outString.toString();
    }

    public native String stringFromJNI();
    public native void testGemm();
    public native void testMatTranspose(String kernelCode);
    public native void testVectorSum(String kernelCode);
    public native void sgemm(String kernelCode);

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }
}
