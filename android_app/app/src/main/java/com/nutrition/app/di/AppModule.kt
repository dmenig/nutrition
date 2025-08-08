package com.nutrition.app.di

import android.content.Context
import androidx.room.Room
import com.nutrition.app.data.local.database.NutritionDatabase
import com.nutrition.app.data.remote.NutritionApi
import com.nutrition.app.data.remote.NutritionApiService
import com.nutrition.app.data.repository.NutritionRepository
import com.nutrition.app.data.NutritionRepository as RemoteNutritionRepository
import com.nutrition.app.data.local.dao.CustomFoodDao
import com.nutrition.app.data.local.dao.FoodLogDao
import com.nutrition.app.data.local.dao.SportActivityDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import java.util.concurrent.TimeUnit
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import javax.inject.Singleton
import androidx.work.WorkManager
import javax.inject.Named

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideNutritionDatabase(@ApplicationContext context: Context): NutritionDatabase {
        return Room.databaseBuilder(
            context.applicationContext,
            NutritionDatabase::class.java,
            "nutrition_database"
        ).build()
    }

    @Provides
    fun provideFoodLogDao(database: NutritionDatabase): FoodLogDao {
        return database.foodLogDao()
    }

    @Provides
    fun provideSportActivityDao(database: NutritionDatabase): SportActivityDao {
        return database.sportActivityDao()
    }

    @Provides
    fun provideCustomFoodDao(database: NutritionDatabase): CustomFoodDao {
        return database.customFoodDao()
    }

    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        val logging = HttpLoggingInterceptor().apply {
            setLevel(HttpLoggingInterceptor.Level.BODY)
        }
        return OkHttpClient.Builder()
            .addInterceptor(logging)
            .connectTimeout(2, TimeUnit.SECONDS)
            .readTimeout(2, TimeUnit.SECONDS)
            .callTimeout(3, TimeUnit.SECONDS)
            .retryOnConnectionFailure(false)
            .build()
    }

    @Provides
    @Singleton
    @Named("NutritionApi")
    fun provideRetrofit(okHttpClient: OkHttpClient): Retrofit {
        return Retrofit.Builder()
            .baseUrl("https://nutrition-tbdo.onrender.com/")
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

    }

    @Provides
    @Singleton
    fun provideNutritionApiService(@Named("NutritionApi") retrofit: Retrofit): NutritionApiService {
        return retrofit.create(NutritionApiService::class.java)
    }

    @Provides
    @Singleton
    fun provideWorkManager(@ApplicationContext context: Context): WorkManager {
        return WorkManager.getInstance(context)
    }

    @Provides
    @Singleton
    fun provideNutritionRepository(
        foodLogDao: FoodLogDao,
        sportActivityDao: SportActivityDao,
        customFoodDao: CustomFoodDao,
        apiService: NutritionApiService
    ): NutritionRepository {
        return NutritionRepository(
            foodLogDao = foodLogDao,
            sportActivityDao = sportActivityDao,
            customFoodDao = customFoodDao,
            apiService = apiService
        )
    }

    @Provides
    @Singleton
    fun provideRemoteNutritionRepository(nutritionApi: NutritionApi): RemoteNutritionRepository {
        return RemoteNutritionRepository(nutritionApi)
    }
}