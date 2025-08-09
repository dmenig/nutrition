package com.nutrition.app.ui.dailylog

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
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
    onSportLogClick: (SportActivity) -> Unit = {},
    onNavigateToFoodEntry: () -> Unit = {},
    onNavigateToSportEntry: () -> Unit = {}
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
        Box(modifier = modifier.fillMaxSize().padding(paddingValues)) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
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
                        Text("Protein: ${dailySummary?.totalProtein ?: "N/A"} g")
                        Text("Carbs: ${dailySummary?.totalCarbs ?: "N/A"} g")
                        Text("Fat: ${dailySummary?.totalFat ?: "N/A"} g")
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
                                val qtyInGrams = if (log.unit.lowercase() == "100g") log.quantity * 100.0 else log.quantity
                                val unitLabel = "g"
                                val details = if (qtyInGrams > 0.0) {
                                    "${"%.0f".format(log.calories)} kcal • ${"%.0f".format(qtyInGrams)} $unitLabel"
                                } else {
                                    "${"%.0f".format(log.calories)} kcal"
                                }
                                Text(text = details, style = MaterialTheme.typography.bodySmall)
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
                            Column(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(16.dp),
                                verticalArrangement = Arrangement.Bottom,
                                horizontalAlignment = Alignment.End
                            ) {
                                FloatingActionButton(onClick = onNavigateToFoodEntry) {
                                    Text("Add Food")
                                }
                                Spacer(modifier = Modifier.height(16.dp))
                                FloatingActionButton(onClick = onNavigateToSportEntry) {
                                    Text("Add Sport")
                                }
                            }
                        }
                    }
                }
            }
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                verticalArrangement = Arrangement.Bottom,
                horizontalAlignment = Alignment.End
            ) {
                FloatingActionButton(onClick = onNavigateToFoodEntry) {
                    Text("Add Food")
                }
                Spacer(modifier = Modifier.height(16.dp))
                FloatingActionButton(onClick = onNavigateToSportEntry) {
                    Text("Add Sport")
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