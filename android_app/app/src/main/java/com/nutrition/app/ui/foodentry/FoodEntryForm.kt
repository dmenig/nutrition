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
import com.nutrition.app.data.local.dao.FrequentFoodItem
import com.nutrition.app.data.local.entities.CustomFood
import com.nutrition.app.data.model.Product
import java.util.Date
import android.content.Intent
import androidx.compose.ui.platform.LocalContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FoodEntryForm(
    modifier: Modifier = Modifier,
    foodName: String = "",
    quantity: String = "",
    unit: String = "",
    loggedAt: Date = Date(),
    onSave: (String, String, String, Date) -> Unit,
    onCancel: () -> Unit,
) {
    var currentFoodName by remember { mutableStateOf(foodName) }
    var currentQuantity by remember { mutableStateOf(quantity) }
    var currentUnit by remember { mutableStateOf(unit) }
    var currentLoggedAt by remember { mutableStateOf(loggedAt) }

    val foodSearchViewModel: FoodSearchViewModel = hiltViewModel()
    val searchQuery by foodSearchViewModel.searchQuery.collectAsState()
    val searchResults by foodSearchViewModel.searchResults.collectAsState()
    var showFoodSearchDialog by remember { mutableStateOf(false) }

    val recentFoods by foodSearchViewModel.recentFoods.collectAsState()
    val frequentFoods by foodSearchViewModel.frequentFoods.collectAsState()

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
                    IconButton(onClick = { showFoodSearchDialog = true }) {
                        Icon(Icons.Filled.Search, "Search Foods")
                    }
                }
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = currentQuantity,
                onValueChange = { currentQuantity = it },
                label = { Text("Quantity") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = currentUnit,
                onValueChange = { currentUnit = it },
                label = { Text("Unit") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(16.dp))
            // Date/Time Picker Placeholder
            Text("Logged At: ${currentLoggedAt.toLocaleString()}")
            Spacer(modifier = Modifier.height(16.dp))

            if (showFoodSearchDialog) {
                AlertDialog(
                    onDismissRequest = { showFoodSearchDialog = false },
                    title = { Text("Select Food") },
                    text = {
                        Column {
                            if (recentFoods.isNotEmpty()) {
                                Text("Recent Foods", style = MaterialTheme.typography.titleMedium)
                                LazyColumn(modifier = Modifier.heightIn(max = 150.dp)) {
                                    items(recentFoods) { foodName ->
                                        Text(
                                            text = foodName,
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .clickable {
                                                    currentFoodName = foodName
                                                    showFoodSearchDialog = false
                                                }
                                                .padding(8.dp)
                                        )
                                    }
                                }
                                Spacer(modifier = Modifier.height(16.dp))
                            }

                            if (frequentFoods.isNotEmpty()) {
                                Text("Frequent Foods", style = MaterialTheme.typography.titleMedium)
                                LazyColumn(modifier = Modifier.heightIn(max = 150.dp)) {
                                    items(frequentFoods) { foodItem ->
                                        Text(
                                            text = "${foodItem.foodName} (${foodItem.foodCount})",
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .clickable {
                                                    currentFoodName = foodItem.foodName
                                                    showFoodSearchDialog = false
                                                }
                                                .padding(8.dp)
                                        )
                                    }
                                }
                                Spacer(modifier = Modifier.height(16.dp))
                            }

                            Text("Search Results", style = MaterialTheme.typography.titleMedium)
                            LazyColumn(modifier = Modifier.heightIn(max = 250.dp)) {
                                items(searchResults) { product ->
                                    Text(
                                        text = product.product_name ?: "Unknown",
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .clickable {
                                                currentFoodName = product.product_name ?: ""
                                                showFoodSearchDialog = false
                                            }
                                            .padding(8.dp)
                                    )
                                }
                            }
                        }
                    },
                    confirmButton = {
                        Button(onClick = { showFoodSearchDialog = false }) {
                            Text("Close")
                        }
                    }
                )
            }

            val context = LocalContext.current
            Button(
                onClick = {
                    val intent = Intent(context, BarcodeScanActivity::class.java)
                    context.startActivity(intent)
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Scan Barcode")
            }
            Spacer(modifier = Modifier.height(16.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                Button(onClick = onCancel) {
                    Text("Cancel")
                }
                Spacer(modifier = Modifier.width(8.dp))
                Button(onClick = { onSave(currentFoodName, currentQuantity, currentUnit, currentLoggedAt) }) {
                    Text("Save")
                }
            }
        }
    }
}