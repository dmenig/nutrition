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
        // Auto-auth on startup with device-scoped credentials
        CoroutineScope(Dispatchers.IO).launch {
            val email = "device@nutrition.app"
            val password = "device-secret"
            authRepository.ensureToken(email, password)
        }
    }
}