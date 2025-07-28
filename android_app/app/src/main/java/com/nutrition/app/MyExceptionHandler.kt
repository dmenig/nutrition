package com.nutrition.app

import android.content.Context
import android.util.Log
import kotlin.system.exitProcess

class MyExceptionHandler(private val context: Context) : Thread.UncaughtExceptionHandler {

    override fun uncaughtException(thread: Thread, throwable: Throwable) {
        Log.e("MyExceptionHandler", "Uncaught exception: ", throwable)
        exitProcess(1)
    }
}