package com.nutrition.app

import android.app.Application
import androidx.hilt.work.HiltWorkerFactory
import androidx.work.Configuration
import dagger.hilt.android.HiltAndroidApp
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import android.util.Log
import com.nutrition.app.data.AuthRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import android.provider.Settings

@HiltAndroidApp
class NutritionApplication : Application(), Configuration.Provider {

    @Inject
    lateinit var workerFactory: HiltWorkerFactory
    @Inject
    lateinit var authRepository: AuthRepository

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder()
            .setWorkerFactory(workerFactory)
            .build()

    override fun onCreate() {
        super.onCreate()
        Log.d("NutritionApplication", "onCreate: Application created")
        Thread.setDefaultUncaughtExceptionHandler(MyExceptionHandler(this))
        // Auto-auth on startup with device-unique credentials to avoid collisions
        CoroutineScope(Dispatchers.IO).launch {
            val androidId = try {
                Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID)
            } catch (_: Exception) { null }
            val suffix = (androidId ?: "unknown").lowercase()
            val email = "device-$suffix@nutrition.app"
            val password = "device-$suffix-secret"
            authRepository.ensureToken(email, password)
        }
    }
}