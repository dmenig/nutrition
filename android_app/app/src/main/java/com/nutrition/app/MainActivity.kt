package com.nutrition.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.NavType
import androidx.navigation.navArgument
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import android.content.Intent
import androidx.compose.ui.platform.LocalContext
import androidx.work.ExistingPeriodicWorkPolicy
import com.nutrition.app.ui.foodentry.BarcodeScanActivity
import com.nutrition.app.ui.foodentry.FoodEntryForm
import com.nutrition.app.ui.foodentry.FoodEntryRoute
import com.nutrition.app.ui.dailylog.DailyLogScreen
import com.nutrition.app.ui.sportentry.SportEntryForm
import com.nutrition.app.ui.sportentry.SportEntryRoute
import com.nutrition.app.ui.theme.NutritionTheme
import com.nutrition.app.ui.customfood.CustomFoodListScreen
import com.nutrition.app.ui.customfood.CustomFoodEntryScreen
import java.util.Date

import androidx.work.Constraints
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import com.nutrition.app.sync.SyncWorker
import dagger.hilt.android.AndroidEntryPoint
import java.util.concurrent.TimeUnit

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()

        val syncWorkRequest = PeriodicWorkRequestBuilder<SyncWorker>(
            15, TimeUnit.MINUTES
        )
            .setConstraints(constraints)
            .build()

        WorkManager.getInstance(applicationContext).enqueueUniquePeriodicWork(
            "SyncWork",
            ExistingPeriodicWorkPolicy.KEEP,
            syncWorkRequest
        )

        setContent {
            NutritionTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    NutritionApp()
                }
            }
        }
    }
}

@Composable
fun NutritionApp() {
    val navController = rememberNavController()
    NavHost(navController = navController, startDestination = "daily_log") {
        composable("daily_log") {
            DailyLogScreen(
                onFoodLogClick = { foodLog ->
                    navController.navigate("food_entry/${foodLog.foodName}/1/g/${Date().time}")
                },
                onSportLogClick = { sportLog ->
                    navController.navigate("sport_entry_route?activityName=${sportLog.activityName}&duration=${sportLog.durationMinutes}")
                },
                onNavigateToFoodEntry = { navController.navigate("food_entry_route") },
                onNavigateToSportEntry = { navController.navigate("sport_entry_route") }
            )
        }
        composable("food_entry_route") {
            FoodEntryRoute(
                onSave = { navController.popBackStack() },
                onCancel = { navController.popBackStack() }
            )
        }
        composable(
            "sport_entry_route?activityName={activityName}&duration={duration}",
            arguments = listOf(
                navArgument("activityName") { type = NavType.StringType; nullable = true },
                navArgument("duration") { type = NavType.StringType; nullable = true }
            )
        ) { backStackEntry ->
            val activityName = backStackEntry.arguments?.getString("activityName")
            val duration = backStackEntry.arguments?.getString("duration")
            SportEntryRoute(
                activityName = activityName,
                duration = duration,
                onSave = { navController.popBackStack() },
                onCancel = { navController.popBackStack() }
            )
        }
        composable("food_entry/{foodName}/{quantity}/{unit}/{loggedAt}") { backStackEntry ->
            val foodName = backStackEntry.arguments?.getString("foodName") ?: ""
            val quantity = backStackEntry.arguments?.getString("quantity") ?: ""
            val unit = backStackEntry.arguments?.getString("unit") ?: ""
            val loggedAt = backStackEntry.arguments?.getString("loggedAt")?.toLongOrNull()?.let { Date(it) } ?: Date()

            FoodEntryForm(
                foodName = foodName,
                quantity = quantity,
                loggedAt = loggedAt,
                onSave = { _, _, _ -> navController.popBackStack() },
                onCancel = { navController.popBackStack() }
            )
        }
        composable("sport_entry/{activityName}/{duration}") { backStackEntry ->
            val activityName = backStackEntry.arguments?.getString("activityName") ?: ""
            val duration = backStackEntry.arguments?.getString("duration") ?: ""

            SportEntryRoute(
                activityName = activityName,
                duration = duration,
                onSave = { navController.popBackStack() },
                onCancel = { navController.popBackStack() }
            )
        }
        composable("custom_food_list") {
            CustomFoodListScreen(
                onAddFoodClick = { navController.navigate("custom_food_entry") }
            )
        }
        composable("custom_food_entry") {
            CustomFoodEntryScreen(
                onBackClick = { navController.popBackStack() }
            )
        }
    }
}