package com.nutrition.app.sync

import android.content.Context
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.nutrition.app.data.local.dao.CustomFoodDao
import com.nutrition.app.data.local.dao.FoodLogDao
import com.nutrition.app.data.local.dao.SportActivityDao
import com.nutrition.app.data.remote.NutritionApiService
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@HiltWorker
class SyncWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted workerParams: WorkerParameters,
    private val foodLogDao: FoodLogDao,
    private val sportActivityDao: SportActivityDao,
    private val customFoodDao: CustomFoodDao,
    private val nutritionApiService: NutritionApiService
) : CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        try {
            val unsyncedFoodLogs = foodLogDao.getUnsyncedFoodLogs()
            unsyncedFoodLogs.forEach { log ->
                try {
                    val response = nutritionApiService.createFoodLog(log)
                    if (response.isSuccessful) {
                        foodLogDao.markFoodLogAsSynced(log.id)
                    } else {
                        // Handle conflict: last-write-wins
                        val serverLog = nutritionApiService.getFoodLog(log.id).body()
                        if (serverLog != null && log.lastModified > serverLog.lastModified) {
                            nutritionApiService.updateFoodLog(log.id, log)
                            foodLogDao.markFoodLogAsSynced(log.id)
                        }
                    }
                } catch (e: Exception) {
                    // Log error, but continue with other items
                }
            }

            val unsyncedSportActivities = sportActivityDao.getUnsyncedSportActivities()
            unsyncedSportActivities.forEach { activity ->
                try {
                    val response = nutritionApiService.createSportActivity(activity)
                    if (response.isSuccessful) {
                        sportActivityDao.markSportActivityAsSynced(activity.id)
                    } else {
                        // Handle conflict: last-write-wins
                        val serverActivity = nutritionApiService.getSportActivity(activity.id).body()
                        if (serverActivity != null && activity.lastModified > serverActivity.lastModified) {
                            nutritionApiService.updateSportActivity(activity.id, activity)
                            sportActivityDao.markSportActivityAsSynced(activity.id)
                        }
                    }
                } catch (e: Exception) {
                    // Log error, but continue with other items
                }
            }

            val unsyncedCustomFoods = customFoodDao.getUnsyncedCustomFoods()
            unsyncedCustomFoods.forEach { customFood ->
                try {
                    val response = nutritionApiService.createCustomFood(customFood)
                    if (response.isSuccessful) {
                        customFoodDao.markCustomFoodAsSynced(customFood.id)
                    } else {
                        // Handle conflict: last-write-wins
                        val serverCustomFood = nutritionApiService.getCustomFood(customFood.id).body()
                        if (serverCustomFood != null && customFood.lastModified > serverCustomFood.lastModified) {
                            nutritionApiService.updateCustomFood(customFood.id, customFood)
                            customFoodDao.markCustomFoodAsSynced(customFood.id)
                        }
                    }
                } catch (e: Exception) {
                    // Log error, but continue with other items
                }
            }

            Result.success()
        } catch (e: Exception) {
            Result.retry()
        }
    }
}