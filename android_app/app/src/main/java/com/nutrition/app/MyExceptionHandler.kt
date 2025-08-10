package com.nutrition.app

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast
import com.nutrition.app.util.ErrorReporter
import kotlin.system.exitProcess

class MyExceptionHandler(private val context: Context) : Thread.UncaughtExceptionHandler {

    override fun uncaughtException(thread: Thread, throwable: Throwable) {
        Log.e("MyExceptionHandler", "Uncaught exception: ", throwable)

        // Best-effort user-visible message before crash
        try {
            Handler(Looper.getMainLooper()).post {
                Toast.makeText(
                    context.applicationContext,
                    "The app encountered an unexpected error and will close.",
                    Toast.LENGTH_LONG
                ).show()
            }
        } catch (_: Exception) {
            // ignored
        }

        // Also route a message through the global reporter for UI listeners
        ErrorReporter.show("Unexpected error: ${throwable.message ?: throwable::class.java.simpleName}")

        // Give the Toast a brief moment to display
        try { Thread.sleep(1500) } catch (_: InterruptedException) {}

        exitProcess(1)
    }
}