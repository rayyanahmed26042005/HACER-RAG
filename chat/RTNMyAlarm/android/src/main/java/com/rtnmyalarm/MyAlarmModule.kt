package com.rtnmyalarm

import android.content.Intent
import android.provider.AlarmClock
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.Promise
import android.util.Log

class MyAlarmModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = NAME

    @ReactMethod
    fun setAlarm(hour: Double, minute: Double, days: ReadableArray, promise: Promise) {
        try {
            // Validate inputs
            if (hour < 0 || hour > 23) {
                promise.reject("INVALID_HOUR", "Hour must be between 0 and 23")
                return
            }
            if (minute < 0 || minute > 59) {
                promise.reject("INVALID_MINUTE", "Minute must be between 0 and 59")
                return
            }

            val dayList = ArrayList<Int>()
            for (i in 0 until days.size()) {
                val day = days.getInt(i)
                if (day < 1 || day > 7) {
                    promise.reject("INVALID_DAY", "Days must be between 1 and 7 (Sunday=1, Saturday=7)")
                    return
                }
                dayList.add(day)
            }

            Log.d("MyAlarmModule", "Setting alarm: hour=${hour.toInt()}, minute=${minute.toInt()}, days=$dayList")

            // Use the same intent configuration as the reference code
            val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
                putExtra(AlarmClock.EXTRA_HOUR, hour.toInt())
                putExtra(AlarmClock.EXTRA_MINUTES, minute.toInt())
                putExtra(AlarmClock.EXTRA_MESSAGE, "Wake up")
                putExtra(AlarmClock.EXTRA_DAYS, dayList)
                putExtra(AlarmClock.EXTRA_SKIP_UI, true)
            }
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK

            if (intent.resolveActivity(reactApplicationContext.packageManager) != null) {
                try {
                    reactApplicationContext.startActivity(intent)
                    Log.d("MyAlarmModule", "AlarmClock intent launched successfully")
                    promise.resolve("Alarm set successfully")
                } catch (e: Exception) {
                    Log.e("MyAlarmModule", "Failed to start activity: ${e.message}")
                    promise.reject("ACTIVITY_ERROR", "Failed to launch AlarmClock intent: ${e.message}")
                }
            } else {
                Log.e("MyAlarmModule", "No compatible clock app found")
                promise.reject(
                    "NO_COMPATIBLE_APP",
                    "No compatible clock app found. Please ensure Google Clock is enabled or install another app that supports AlarmClock intents."
                )
            }
        } catch (e: Exception) {
            Log.e("MyAlarmModule", "Unexpected error in setAlarm: ${e.message}")
            promise.reject("UNEXPECTED_ERROR", "Unexpected error: ${e.message}")
        }
    }

    companion object {
        const val NAME = "RTNMyAlarm"
    }
}