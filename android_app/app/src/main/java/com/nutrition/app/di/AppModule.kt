package com.nutrition.app.di

import android.content.Context
import androidx.room.Room
import com.nutrition.app.data.local.database.NutritionDatabase
import com.nutrition.app.data.local.database.MIGRATION_1_2
import com.nutrition.app.data.local.database.MIGRATION_2_3
import com.nutrition.app.data.local.database.MIGRATION_3_4
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
import okhttp3.Dns
import okhttp3.logging.HttpLoggingInterceptor
import java.util.concurrent.TimeUnit
import java.net.Inet4Address
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import javax.inject.Singleton
import androidx.work.WorkManager
import javax.inject.Named
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonDeserializationContext
import com.google.gson.JsonDeserializer
import com.google.gson.JsonElement
import com.google.gson.JsonPrimitive
import com.google.gson.JsonSerializationContext
import com.google.gson.JsonSerializer
import java.lang.reflect.Type
import java.time.Instant
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import com.nutrition.app.util.ErrorReportingInterceptor

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
        )
            .addMigrations(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4)
            .build()
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
        // Prefer IPv4 addresses to avoid potential broken IPv6 paths on some networks
        val ipv4OnlyDns = object : Dns {
            override fun lookup(hostname: String): List<java.net.InetAddress> {
                return try {
                    Dns.SYSTEM.lookup(hostname).filterIsInstance<Inet4Address>()
                } catch (_: Exception) {
                    emptyList()
                }
            }
        }
        return OkHttpClient.Builder()
            .addInterceptor(logging)
            .addInterceptor(ErrorReportingInterceptor())
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            // No overall call timeout to allow slow cold starts on server
            .dns(ipv4OnlyDns)
            .retryOnConnectionFailure(true)
            .protocols(listOf(okhttp3.Protocol.HTTP_1_1))
            .build()
    }

    @Provides
    @Singleton
    fun provideGson(): Gson {
        val localDateAdapter = object : JsonSerializer<LocalDate>, JsonDeserializer<LocalDate> {
            override fun serialize(src: LocalDate?, typeOfSrc: Type?, context: JsonSerializationContext?): JsonElement {
                return JsonPrimitive(src?.toString())
            }

            override fun deserialize(json: JsonElement?, typeOfT: Type?, context: JsonDeserializationContext?): LocalDate {
                return LocalDate.parse(json!!.asString)
            }
        }

        val localDateTimeAdapter = object : JsonSerializer<LocalDateTime>, JsonDeserializer<LocalDateTime> {
            override fun serialize(src: LocalDateTime?, typeOfSrc: Type?, context: JsonSerializationContext?): JsonElement {
                // Serialize as ISO instant in UTC
                val instant = src!!.atZone(ZoneOffset.UTC).toInstant()
                return JsonPrimitive(DateTimeFormatter.ISO_INSTANT.format(instant))
            }

            override fun deserialize(json: JsonElement?, typeOfT: Type?, context: JsonDeserializationContext?): LocalDateTime {
                val value = json!!.asString
                return try {
                    // Handle strings with timezone like ...Z
                    val instant = Instant.parse(value)
                    LocalDateTime.ofInstant(instant, ZoneOffset.UTC)
                } catch (_: Exception) {
                    // Fallback to local date-time without zone
                    LocalDateTime.parse(value, DateTimeFormatter.ISO_LOCAL_DATE_TIME)
                }
            }
        }

        return GsonBuilder()
            .registerTypeAdapter(LocalDate::class.java, localDateAdapter)
            .registerTypeAdapter(LocalDateTime::class.java, localDateTimeAdapter)
            .create()
    }

    @Provides
    @Singleton
    @Named("NutritionApi")
    fun provideRetrofit(okHttpClient: OkHttpClient, gson: Gson): Retrofit {
        return Retrofit.Builder()
            .baseUrl("https://nutrition-tbdo.onrender.com/")
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create(gson))
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