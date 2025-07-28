package com.nutrition.app.ui.sportentry

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import java.util.Date

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SportEntryForm(
    modifier: Modifier = Modifier,
    activityName: String = "",
    duration: String = "",
    caloriesExpended: String = "",
    onSave: (String, String, String) -> Unit,
    onCancel: () -> Unit
) {
    var currentActivityName by remember { mutableStateOf(activityName) }
    var currentDuration by remember { mutableStateOf(duration) }
    var currentCaloriesExpended by remember { mutableStateOf(caloriesExpended) }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Sport Entry") })
        }
    ) { paddingValues ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
        ) {
            OutlinedTextField(
                value = currentActivityName,
                onValueChange = { currentActivityName = it },
                label = { Text("Activity Name") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = currentDuration,
                onValueChange = { currentDuration = it },
                label = { Text("Duration (minutes)") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = currentCaloriesExpended,
                onValueChange = { currentCaloriesExpended = it },
                label = { Text("Calories Expended") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(16.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                Button(onClick = onCancel) {
                    Text("Cancel")
                }
                Spacer(modifier = Modifier.width(8.dp))
                Button(onClick = { onSave(currentActivityName, currentDuration, currentCaloriesExpended) }) {
                    Text("Save")
                }
            }
        }
    }
}