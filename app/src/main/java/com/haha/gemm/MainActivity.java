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

public class MainActivity extends Activity {
    private static final int ARRAY_SIZE = 512*512;
    private HandlerThread mHandlerThread = new HandlerThread("test-gemm");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mHandlerThread.start();

        // Example of a call to a native method
        TextView tv = (TextView) findViewById(R.id.sample_text);
        tv.setText(stringFromJNI());

        findViewById(R.id.id_test_gemm).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Handler handler = new Handler(mHandlerThread.getLooper());
                handler.post(new Runnable() {
                    @Override
                    public void run() {
//                        testGemm();

                        for(int i = 0; i < 1; ++i){
                            //calcVectorsNative(null);
                            calcVectors(null);
                        }
                    }
                });
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mHandlerThread.quitSafely();
    }

    public void calcVectors(View view)
    {
        int[] arrayA = new int[ARRAY_SIZE];
        int[] arrayB = new int[ARRAY_SIZE];
        int[] arrayC = new int[ARRAY_SIZE];
        float[] execTime = new float[1];
        execTime[0] = 0;

        AssetManager am = getAssets();

        try
        {
            initArrays(arrayA, arrayB, arrayC, ARRAY_SIZE);
            InputStream is = am.open("OpenCLFirstApp_kernel.cl");
            String kernelCode = convertInputStreamToString(is);

            addArraysViaOpenCL(arrayA,arrayB,arrayC, kernelCode, execTime);
        }
        catch(IOException e)
        {
            Log.d("oclDebug",e.toString());
        }

        String print = String.valueOf(execTime[0]);
        print += " (ms)";
//        TextView myTextField = (TextView)findViewById(R.id.sample_text);
//        myTextField.setText(print);
        Log.d("RoclDebug", "result = " + print);
    }

    public void calcVectorsNative(View view)
    {
        int[] arrayA = new int[ARRAY_SIZE];
        int[] arrayB = new int[ARRAY_SIZE];
        int[] arrayC = new int[ARRAY_SIZE];

        long difference=0;

        initArrays(arrayA, arrayB, arrayC, ARRAY_SIZE);

        long startTime = System.currentTimeMillis();
        calcArrays(arrayA, arrayB, arrayC, ARRAY_SIZE);
        difference = System.currentTimeMillis() - startTime;

        String print = String.valueOf(difference);
        print += " (ms)";
        Log.d("RoclDebug", "result = " + print);
//        TextView myTextField = (TextView)findViewById(R.id.sample_text);
//        myTextField.setText(print);
    }

    public String convertInputStreamToString(InputStream in)
    {
        BufferedReader br;
        StringBuffer outString = new StringBuffer();
        br = new BufferedReader(new InputStreamReader(in));
        try{
            String read = br.readLine();

            while(read != null)
            {
                outString.append(read);
                read = br.readLine();
            }
        }
        catch(IOException e)
        {
            Log.d("oclDebug",e.toString());
        }

        return outString.toString();
    }

    public void initArrays(int[] arrayA, int[] arrayB, int[] arrayC, int size)
    {
        for(int i=0 ; i<size ; ++i)
        {
            arrayA[i] = i;
            arrayB[i] = size - i;
            arrayC[i] = 0;
        }
    }

    public void calcArrays(int[] arrayA, int[] arrayB, int[] arrayC, int size)
    {
        for(int i=0 ; i<size ; ++i)
        {
            arrayC[i] = arrayB[i] + arrayA[i];
        }
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
    public native void testGemm();
    public native void  addArraysViaOpenCL(int[] arrayA,
                                           int[] arrayB,
                                           int[] arrayC,
                                           String kernelCode,
                                           float[] runTime);

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }
}
