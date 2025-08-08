# Real-Time Food Suggestions

This document describes the real-time food suggestion feature, including the new API endpoint for food autocompletion.

## API Endpoint: `/api/v1/foods/autocomplete`

### Description
This endpoint provides real-time food suggestions based on a partial query string. It allows users to quickly find food items by typing a few characters of the food name. The search is case-insensitive and returns a limited number of matching results.

### Request
- **Method**: `GET`
- **URL**: `/api/v1/foods/autocomplete`
- **Query Parameters**:
    - `q` (string, required): The partial food name to search for.
      - Minimum length: 1 character.

### Example Request
```
GET /api/v1/foods/autocomplete?q=appl
```

### Response
- **Content Type**: `application/json`
- **Body**: A JSON array of food objects that partially match the query. Each food object includes:
    - `id` (integer): The unique identifier of the food.
    - `name` (string): The name of the food.
    - `calories` (float): The calorie count of the food.
    - `protein` (float): The protein content of the food.
    - `carbs` (float): The carbohydrate content of the food.
    - `fat` (float): The fat content of the food.

### Example Response
```json
[
  {
    "id": 1,
    "name": "Apple",
    "calories": 52.0,
    "protein": 0.3,
    "carbs": 14.0,
    "fat": 0.2
  },
  {
    "id": 5,
    "name": "Applesauce",
    "calories": 42.0,
    "protein": 0.2,
    "carbs": 11.0,
    "fat": 0.1
  }
]