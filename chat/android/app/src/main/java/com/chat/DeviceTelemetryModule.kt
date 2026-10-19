package com.chat

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import java.io.File
import java.util.Locale

/** Read-only device signals used by the opt-in inference controller. */
class DeviceTelemetryModule(private val context: ReactApplicationContext) :
    ReactContextBaseJavaModule(context) {

    override fun getName(): String = "DeviceTelemetryModule"

    @ReactMethod
    fun getSnapshot(promise: Promise) {
        try {
            val result = Arguments.createMap()
            result.putInt("availableProcessors", Runtime.getRuntime().availableProcessors())

            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val memoryInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memoryInfo)
            result.putDouble("availableMemoryMb", memoryInfo.availMem / (1024.0 * 1024.0))
            result.putDouble("totalMemoryMb", memoryInfo.totalMem / (1024.0 * 1024.0))
            result.putBoolean("lowMemory", memoryInfo.lowMemory)

            val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                result.putInt("thermalStatus", powerManager.currentThermalStatus)
                result.putString("thermalStatusName", thermalStatusName(powerManager.currentThermalStatus))
            } else {
                result.putInt("thermalStatus", -1)
                result.putString("thermalStatusName", "unavailable")
            }

            val batteryIntent = context.registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            if (batteryIntent != null) {
                val level = batteryIntent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = batteryIntent.getIntExtra(BatteryManager.EXTRA_SCALE, 100)
                val temperature = batteryIntent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1)
                if (level >= 0 && scale > 0) result.putDouble("batteryLevel", level * 100.0 / scale)
                if (temperature >= 0) result.putDouble("batteryTempC", temperature / 10.0)
            }

            val frequencies = Arguments.createArray()
            cpuFrequencyFiles().forEach { file ->
                val value = file.readText().trim().toLongOrNull()
                if (value != null) frequencies.pushDouble(value.toDouble())
            }
            result.putArray("cpuFrequenciesKhz", frequencies)
            promise.resolve(result)
        } catch (error: Exception) {
            promise.reject("DEVICE_TELEMETRY_ERROR", "Unable to read device state", error)
        }
    }

    private fun cpuFrequencyFiles(): List<File> {
        val cpuRoot = File("/sys/devices/system/cpu")
        return cpuRoot.listFiles()
            ?.filter { it.name.matches(Regex("cpu[0-9]+")) }
            ?.sortedBy { it.name.removePrefix("cpu").toIntOrNull() ?: Int.MAX_VALUE }
            ?.map { File(it, "cpufreq/scaling_cur_freq") }
            ?.filter { it.canRead() }
            ?: emptyList()
    }

    private fun thermalStatusName(status: Int): String = when (status) {
        PowerManager.THERMAL_STATUS_NONE -> "none"
        PowerManager.THERMAL_STATUS_LIGHT -> "light"
        PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
        PowerManager.THERMAL_STATUS_SEVERE -> "severe"
        PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
        else -> String.format(Locale.US, "unknown_%d", status)
    }
}
