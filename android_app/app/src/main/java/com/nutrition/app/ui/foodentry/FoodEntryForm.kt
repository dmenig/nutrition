package com.nutrition.app.ui.foodentry

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.nutrition.app.data.model.Food
import java.time.LocalDateTime
import java.util.Date
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FoodEntryForm(
    modifier: Modifier = Modifier,
    foodName: String = "",
    quantity: String = "",
    loggedAt: Date = Date(),
    onSave: (String, String, Date) -> Unit,
    onCancel: () -> Unit
) {
    var currentFoodName by remember { mutableStateOf(foodName) }
    var currentQuantity by remember { mutableStateOf(quantity) }
    var currentLoggedAt by remember { mutableStateOf(loggedAt) }

    val foodSearchViewModel: FoodSearchViewModel = hiltViewModel()
    val searchQuery by foodSearchViewModel.searchQuery.collectAsState()
    val searchResults by foodSearchViewModel.searchResults.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Food Entry") })
        }
    ) { paddingValues ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
        ) {
            OutlinedTextField(
                value = currentFoodName,
                onValueChange = {
                    currentFoodName = it
                    foodSearchViewModel.onSearchQueryChanged(it)
                },
                label = { Text("Food Name (Search or Custom)") },
                modifier = Modifier.fillMaxWidth(),
                trailingIcon = {
                    if (currentFoodName.isNotBlank()) {
                        IconButton(onClick = { currentFoodName = "" }) {
                            Icon(Icons.Filled.Search, "Clear Search")
                        }
                    }
                }
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = currentQuantity,
                onValueChange = { input ->
                    // Allow only digits and optional decimal point
                    val sanitized = input.replace("[^0-9.]".toRegex(), "")
                    currentQuantity = sanitized
                },
                label = { Text("Quantity (g)") },
                isError = currentQuantity.isBlank() || (currentQuantity.toFloatOrNull() ?: 0f) <= 0f,
                supportingText = {
                    if (currentQuantity.isBlank() || (currentQuantity.toFloatOrNull() ?: 0f) <= 0f) {
                        Text("Enter a quantity > 0")
                    }
                },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            // Date/Time Picker Placeholder
            Text("Logged At: ${currentLoggedAt.toLocaleString()}")
            Spacer(modifier = Modifier.height(16.dp))

            if (searchQuery.isNotBlank()) {
                Spacer(modifier = Modifier.height(16.dp))
                Column {
                    if (searchResults.isNotEmpty() && searchQuery.isNotBlank()) {
                        Text("Search Results", style = MaterialTheme.typography.titleMedium)
                        LazyColumn(modifier = Modifier.heightIn(max = 250.dp)) {
                            items(searchResults) { food: Food ->
                                Text(
                                    text = food.name,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable {
                                            foodSearchViewModel.onSuggestionSelected(food)
                                            currentFoodName = food.name
                                        }
                                        .padding(8.dp)
                                )
                            }
                        }
                    }
                }
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                Button(onClick = onCancel) {
                    Text("Cancel")
                }
                Spacer(modifier = Modifier.width(8.dp))
                Button(onClick = {
                    if ((currentQuantity.toFloatOrNull() ?: 0f) <= 0f) return@Button
                    onSave(
                        currentFoodName,
                        currentQuantity,
                        currentLoggedAt
                    )
                }) {
                    Text("Save")
                }
            }
        }
    }
}