package com.nutrition.app.ui.customfood

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.nutrition.app.data.remote.model.BackendFood

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CustomFoodListScreen(
    onAddFoodClick: () -> Unit,
    viewModel: CustomFoodViewModel = hiltViewModel()
) {
    val foods by viewModel.foods.collectAsState()
    var query by remember { mutableStateOf("") }
    // Prompt user to search (backend requires min length 1)
    LaunchedEffect(query) {
        if (query.isNotBlank()) viewModel.refresh(query)
    }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Custom Foods") })
        },
        floatingActionButton = {
            FloatingActionButton(onClick = onAddFoodClick) {
                Icon(Icons.Filled.Add, "Add new custom food")
            }
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .padding(paddingValues)
                .fillMaxSize()
                .padding(16.dp)
        ) {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                label = { Text("Search foods") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(12.dp))

            if (foods.isEmpty()) {
            Text(
                text = if (query.isBlank()) "Type to search foods to edit" else "No foods found",
                modifier = Modifier
                    .fillMaxSize()
                    .wrapContentSize()
            )
            } else {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxSize()
                ) {
                    items(foods) { food ->
                        CustomFoodItem(food = food, onDeleteClick = {
                            viewModel.deleteFood(food.id)
                            // refresh after delete
                            if (query.isNotBlank()) viewModel.refresh(query)
                        })
                    }
                }
            }
        }
    }
}

@Composable
fun CustomFoodItem(food: BackendFood, onDeleteClick: (BackendFood) -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(text = food.name, style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(4.dp))
                Text(text = "Calories: ${food.calories}kcal", style = MaterialTheme.typography.bodySmall)
                Text(text = "Protein: ${food.protein ?: 0f}g", style = MaterialTheme.typography.bodySmall)
                Text(text = "Carbs: ${food.carbs ?: 0f}g", style = MaterialTheme.typography.bodySmall)
                Text(text = "Fat: ${food.fat ?: 0f}g", style = MaterialTheme.typography.bodySmall)
            }
            IconButton(onClick = { onDeleteClick(food) }) {
                Icon(Icons.Filled.Delete, "Delete custom food")
            }
        }
    }
}