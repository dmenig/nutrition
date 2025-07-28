package com.nutrition.app.ui.customfood

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.nutrition.app.data.model.CustomFood

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CustomFoodEntryScreen(
    onBackClick: () -> Unit,
    viewModel: CustomFoodViewModel = hiltViewModel()
) {
    var name by remember { mutableStateOf("") }
    var calories by remember { mutableStateOf("") }
    var protein by remember { mutableStateOf("") }
    var carbohydrates by remember { mutableStateOf("") }
    var fat by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Add Custom Food") },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(Icons.Filled.ArrowBack, "Back")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .padding(paddingValues)
                .padding(16.dp)
                .fillMaxSize()
        ) {
            OutlinedTextField(
                value = name,
                onValueChange = { name = it },
                label = { Text("Food Name") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = calories,
                onValueChange = { calories = it },
                label = { Text("Calories (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = protein,
                onValueChange = { protein = it },
                label = { Text("Protein (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = carbohydrates,
                onValueChange = { carbohydrates = it },
                label = { Text("Carbohydrates (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = fat,
                onValueChange = { fat = it },
                label = { Text("Fat (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(16.dp))
            Button(
                onClick = {
                    val newFood = CustomFood(
                        name = name,
                        caloriesPer100g = calories.toDoubleOrNull() ?: 0.0,
                        proteinPer100g = protein.toDoubleOrNull() ?: 0.0,
                        carbohydratesPer100g = carbohydrates.toDoubleOrNull() ?: 0.0,
                        fatPer100g = fat.toDoubleOrNull() ?: 0.0
                    )
                    viewModel.addCustomFood(newFood)
                    onBackClick()
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Add Food")
            }
        }
    }
}