package com.nutrition.app.ui.dailylog

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.nutrition.app.data.local.entities.FoodLog
import com.nutrition.app.data.local.entities.SportActivity
import java.util.Date

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DailyLogScreen(
    modifier: Modifier = Modifier,
    viewModel: DailyLogViewModel = hiltViewModel(),
    onFoodLogClick: (FoodLog) -> Unit = {},
    onSportLogClick: (SportActivity) -> Unit = {}
) {
    val selectedDate by viewModel.selectedDate.collectAsState()
    val dailySummary by viewModel.dailySummary.collectAsState()
    val foodLogs by viewModel.foodLogs.collectAsState()
    val sportLogs by viewModel.sportLogs.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Daily Log") })
        }
    ) { paddingValues ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
        ) {
            // Calendar View
            CalendarView(
                selectedDate = selectedDate,
                onDateSelected = { date -> viewModel.selectDate(date) },
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Daily Summary Section (Placeholder for now)
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Daily Summary:")
                    Text("Calories: ${dailySummary?.totalCalories ?: "N/A"}")
                    Text("Protein: ${dailySummary?.totalProtein ?: "N/A"}")
                    Text("Carbs: ${dailySummary?.totalCarbs ?: "N/A"}")
                    Text("Fat: ${dailySummary?.totalFat ?: "N/A"}")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Paginated List of Food and Sport Logs
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(vertical = 8.dp)
            ) {
                item {
                    Text("Food Logs:", style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                }
                items(foodLogs) { log ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp)
                            .clickable { onFoodLogClick(log) }
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text(text = log.foodName, style = MaterialTheme.typography.bodyLarge)
                            Text(text = "${log.calories} kcal", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }

                item {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("Sport Logs:", style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                }
                items(sportLogs) { log ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp)
                            .clickable { onSportLogClick(log) }
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text(text = log.activityName, style = MaterialTheme.typography.bodyLarge)
                            Text(text = "${log.durationMinutes} min", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }
    }
}

// Extension function for Date formatting (will be replaced by a proper formatter)
fun Date.toFormattedString(): String {
    return this.toLocaleString() // Simple formatting for now
}

@Preview(showBackground = true)
@Composable
fun DailyLogScreenPreview() {
    DailyLogScreen()
}