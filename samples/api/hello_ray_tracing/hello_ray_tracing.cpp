/* Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hello_ray_tracing.h"

#include "common/vk_common.h"
#include "core/util/logging.hpp"
#include "filesystem/legacy.h"
#include "platform/window.h"

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
/// @brief A debug callback called from Vulkan validation layers.
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT      message_severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT             message_types,
                                                     const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
                                                     void                                       *user_data)
{
	if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
	{
		LOGE("{} Validation Layer: Error: {}: {}", callback_data->messageIdNumber, callback_data->pMessageIdName, callback_data->pMessage);
	}
	else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
	{
		LOGW("{} Validation Layer: Warning: {}: {}", callback_data->messageIdNumber, callback_data->pMessageIdName, callback_data->pMessage);
	}
	else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
	{
		LOGI("{} Validation Layer: Information: {}: {}", callback_data->messageIdNumber, callback_data->pMessageIdName, callback_data->pMessage);
	}
	else if (message_types & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
	{
		LOGI("{} Validation Layer: Performance warning: {}: {}", callback_data->messageIdNumber, callback_data->pMessageIdName, callback_data->pMessage);
	}
	else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
	{
		LOGD("{} Validation Layer: Verbose: {}: {}", callback_data->messageIdNumber, callback_data->pMessageIdName, callback_data->pMessage);
	}
	return VK_FALSE;
}
#endif

/**
 * @brief Validates a list of required extensions, comparing it with the available ones.
 *
 * @param required A vector containing required extension names.
 * @param available A VkExtensionProperties object containing available extensions.
 * @return true if all required extensions are available
 * @return false otherwise
 */
bool HelloRayTracing::validate_extensions(const std::vector<const char *>          &required,
                                           const std::vector<VkExtensionProperties> &available)
{
	bool all_found = true;

	for (const auto *extension_name : required)
	{
		bool found = false;
		for (const auto &available_extension : available)
		{
			if (strcmp(available_extension.extensionName, extension_name) == 0)
			{
				found = true;
				break;
			}
		}

		if (!found)
		{
			// Output an error message for the missing extension
			LOGE("Error: Required extension not found: {}", extension_name);
			all_found = false;
		}
	}

	return all_found;
}

/**
 * @brief Initializes the Vulkan instance.
 */
void HelloRayTracing::init_instance()
{
	LOGI("Initializing Vulkan instance.");

	if (volkInitialize())
	{
		throw std::runtime_error("Failed to initialize volk.");
	}

	uint32_t instance_extension_count;
	VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr));

	std::vector<VkExtensionProperties> available_instance_extensions(instance_extension_count);
	VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, available_instance_extensions.data()));

	std::vector<const char *> required_instance_extensions{VK_KHR_SURFACE_EXTENSION_NAME};

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
	bool has_debug_utils = false;
	for (const auto &ext : available_instance_extensions)
	{
		if (strncmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME, strlen(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) == 0)
		{
			has_debug_utils = true;
			break;
		}
	}
	if (has_debug_utils)
	{
		required_instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	else
	{
		LOGW("{} is not available; disabling debug utils messenger", VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
#endif

#if (defined(VKB_ENABLE_PORTABILITY))
	required_instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
	bool portability_enumeration_available = false;
	if (std::ranges::any_of(available_instance_extensions,
	                        [](VkExtensionProperties const &extension) { return strcmp(extension.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0; }))
	{
		required_instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		portability_enumeration_available = true;
	}
#endif

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
	required_instance_extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WIN32_KHR)
	required_instance_extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_METAL_EXT)
	required_instance_extensions.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
	required_instance_extensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
	required_instance_extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
	required_instance_extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_DISPLAY_KHR)
	required_instance_extensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
#else
#	pragma error Platform not supported
#endif

	if (!validate_extensions(required_instance_extensions, available_instance_extensions))
	{
		throw std::runtime_error("Required instance extensions are missing.");
	}

	std::vector<const char *> requested_instance_layers{};

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
	char const *validationLayer = "VK_LAYER_KHRONOS_validation";

	uint32_t instance_layer_count;
	VK_CHECK(vkEnumerateInstanceLayerProperties(&instance_layer_count, nullptr));

	std::vector<VkLayerProperties> supported_instance_layers(instance_layer_count);
	VK_CHECK(vkEnumerateInstanceLayerProperties(&instance_layer_count, supported_instance_layers.data()));

	if (std::ranges::any_of(supported_instance_layers, [&validationLayer](auto const &lp) { return strcmp(lp.layerName, validationLayer) == 0; }))
	{
		requested_instance_layers.push_back(validationLayer);
		LOGI("Enabled Validation Layer {}", validationLayer);
	}
	else
	{
		LOGW("Validation Layer {} is not available", validationLayer);
	}
#endif

	VkApplicationInfo app{
	    .sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO,
	    .pApplicationName = "Hello Ray Tracing V1.3",
	    .pEngineName      = "Vulkan Samples",
	    .apiVersion       = VK_API_VERSION_1_3};

	VkInstanceCreateInfo instance_info{
	    .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
	    .pApplicationInfo        = &app,
	    .enabledLayerCount       = vkb::to_u32(requested_instance_layers.size()),
	    .ppEnabledLayerNames     = requested_instance_layers.data(),
	    .enabledExtensionCount   = vkb::to_u32(required_instance_extensions.size()),
	    .ppEnabledExtensionNames = required_instance_extensions.data()};

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
	VkDebugUtilsMessengerCreateInfoEXT debug_messenger_create_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
	if (has_debug_utils)
	{
		debug_messenger_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
		debug_messenger_create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
		debug_messenger_create_info.pfnUserCallback = debug_callback;

		instance_info.pNext = &debug_messenger_create_info;
	}
#endif

#if (defined(VKB_ENABLE_PORTABILITY))
	if (portability_enumeration_available)
	{
		instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
	}
#endif

	// Create the Vulkan instance
	VK_CHECK(vkCreateInstance(&instance_info, nullptr, &context.instance));

	volkLoadInstance(context.instance);

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
	if (has_debug_utils)
	{
		VK_CHECK(vkCreateDebugUtilsMessengerEXT(context.instance, &debug_messenger_create_info, nullptr, &context.debug_callback));
	}
#endif
}

/**
 * @brief Initializes the Vulkan physical device and logical device.
 */
void HelloRayTracing::init_device()
{
	LOGI("Initializing Vulkan device.");

	uint32_t gpu_count = 0;
	VK_CHECK(vkEnumeratePhysicalDevices(context.instance, &gpu_count, nullptr));

	if (gpu_count < 1)
	{
		throw std::runtime_error("No physical device found.");
	}

	std::vector<VkPhysicalDevice> gpus(gpu_count);
	VK_CHECK(vkEnumeratePhysicalDevices(context.instance, &gpu_count, gpus.data()));

	for (const auto &physical_device : gpus)
	{
		// Check if the device supports Vulkan 1.3
		VkPhysicalDeviceProperties device_properties;
		vkGetPhysicalDeviceProperties(physical_device, &device_properties);

		if (device_properties.apiVersion < VK_API_VERSION_1_3)
		{
			LOGW("Physical device '{}' does not support Vulkan 1.3, skipping.", device_properties.deviceName);
			continue;
		}

		// Find a queue family that supports graphics and presentation
		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

		std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_family_properties.data());

		for (uint32_t i = 0; i < queue_family_count; i++)
		{
			VkBool32 supports_present = VK_FALSE;
			vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, context.surface, &supports_present);

			if ((queue_family_properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && supports_present)
			{
				context.graphics_queue_index = i;
				break;
			}
		}

		if (context.graphics_queue_index >= 0)
		{
			context.gpu = physical_device;
			break;
		}
	}

	if (context.graphics_queue_index < 0)
	{
		throw std::runtime_error("Failed to find a suitable GPU with Vulkan 1.3 support.");
	}

	uint32_t device_extension_count;

	VK_CHECK(vkEnumerateDeviceExtensionProperties(context.gpu, nullptr, &device_extension_count, nullptr));

	std::vector<VkExtensionProperties> device_extensions(device_extension_count);

	VK_CHECK(vkEnumerateDeviceExtensionProperties(context.gpu, nullptr, &device_extension_count, device_extensions.data()));

	// Since this sample has visual output, the device needs to support the swapchain extension
	std::vector<const char *> required_device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

	if (!validate_extensions(required_device_extensions, device_extensions))
	{
		throw std::runtime_error("Required device extensions are missing");
	}

#if (defined(VKB_ENABLE_PORTABILITY))
	// VK_KHR_portability_subset must be enabled if present in the implementation (e.g on macOS/iOS with beta extensions enabled)
	if (std::ranges::any_of(device_extensions,
	                        [](VkExtensionProperties const &extension) { return strcmp(extension.extensionName, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) == 0; }))
	{
		required_device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
	}
#endif

	// Query for Vulkan 1.3 features
	VkPhysicalDeviceFeatures2                       query_device_features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	VkPhysicalDeviceVulkan13Features                query_vulkan13_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	VkPhysicalDeviceExtendedDynamicStateFeaturesEXT query_extended_dynamic_state_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
	query_device_features2.pNext  = &query_vulkan13_features;
	query_vulkan13_features.pNext = &query_extended_dynamic_state_features;

	vkGetPhysicalDeviceFeatures2(context.gpu, &query_device_features2);

	// Check if Physical device supports Vulkan 1.3 features
	if (!query_vulkan13_features.dynamicRendering)
	{
		throw std::runtime_error("Dynamic Rendering feature is missing");
	}

	if (!query_vulkan13_features.synchronization2)
	{
		throw std::runtime_error("Synchronization2 feature is missing");
	}

	if (!query_extended_dynamic_state_features.extendedDynamicState)
	{
		throw std::runtime_error("Extended Dynamic State feature is missing");
	}

	// Enable only specific Vulkan 1.3 features

	VkPhysicalDeviceExtendedDynamicStateFeaturesEXT enable_extended_dynamic_state_features = {
	    .sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
	    .extendedDynamicState = VK_TRUE};

	VkPhysicalDeviceVulkan13Features enable_vulkan13_features = {
	    .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
	    .pNext            = &enable_extended_dynamic_state_features,
	    .synchronization2 = VK_TRUE,
	    .dynamicRendering = VK_TRUE,
	};

	VkPhysicalDeviceFeatures2 enable_device_features2{
	    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
	    .pNext = &enable_vulkan13_features};
	// Create the logical device

	float queue_priority = 1.0f;

	// Create one queue
	VkDeviceQueueCreateInfo queue_info{
	    .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
	    .queueFamilyIndex = static_cast<uint32_t>(context.graphics_queue_index),
	    .queueCount       = 1,
	    .pQueuePriorities = &queue_priority};

	VkDeviceCreateInfo device_info{
	    .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
	    .pNext                   = &enable_device_features2,
	    .queueCreateInfoCount    = 1,
	    .pQueueCreateInfos       = &queue_info,
	    .enabledExtensionCount   = vkb::to_u32(required_device_extensions.size()),
	    .ppEnabledExtensionNames = required_device_extensions.data()};

	VK_CHECK(vkCreateDevice(context.gpu, &device_info, nullptr, &context.device));
	volkLoadDevice(context.device);

	vkGetDeviceQueue(context.device, context.graphics_queue_index, 0, &context.queue);
}

/**
 * @brief Initializes the vertex buffer by creating it, allocating memory, binding the memory, and uploading vertex data.
 * @note This function must be called after the Vulkan device has been initialized.
 * @throws std::runtime_error if any Vulkan operation fails.
 */
void HelloRayTracing::init_vertex_buffer()
{
	VkDeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();

	// Create the vertex buffer
	VkBufferCreateInfo vertext_buffer_info{
	    .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	    .flags       = 0,
	    .size        = buffer_size,
	    .usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	    .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

	VK_CHECK(vkCreateBuffer(context.device, &vertext_buffer_info, nullptr, &context.vertex_buffer));

	// Get memory requirements
	VkMemoryRequirements memory_requirements;
	vkGetBufferMemoryRequirements(context.device, context.vertex_buffer, &memory_requirements);

	// Allocate memory for the buffer
	VkMemoryAllocateInfo alloc_info{
	    .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .allocationSize  = memory_requirements.size,
	    .memoryTypeIndex = find_memory_type(context.gpu, memory_requirements.memoryTypeBits,
	                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};

	VK_CHECK(vkAllocateMemory(context.device, &alloc_info, nullptr, &context.vertex_buffer_memory));

	// Bind the buffer with the allocated memory
	VK_CHECK(vkBindBufferMemory(context.device, context.vertex_buffer, context.vertex_buffer_memory, 0));

	// Map the memory and copy the vertex data
	void *data;
	VK_CHECK(vkMapMemory(context.device, context.vertex_buffer_memory, 0, buffer_size, 0, &data));
	memcpy(data, vertices.data(), static_cast<size_t>(buffer_size));
	vkUnmapMemory(context.device, context.vertex_buffer_memory);
}

/**
 * @brief Finds a suitable memory type index for allocating memory.
 *
 * This function searches through the physical device's memory types to find one that matches
 * the requirements specified by `type_filter` and `properties`. It's typically used when allocating
 * memory for buffers or images, ensuring that the memory type supports the desired properties.
 *
 * @param physical_device The Vulkan physical device to query for memory properties.
 * @param type_filter A bitmask specifying the acceptable memory types.
 *                    This is usually obtained from `VkMemoryRequirements::memoryTypeBits`.
 * @param properties A bitmask specifying the desired memory properties,
 *                   such as `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` or `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
 * @return The index of a suitable memory type.
 * @throws std::runtime_error if no suitable memory type is found.
 */
uint32_t HelloRayTracing::find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties)
{
	// Structure to hold the physical device's memory properties
	VkPhysicalDeviceMemoryProperties mem_properties;
	vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

	// Iterate over all memory types available on the physical device
	for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
	{
		// Check if the current memory type is acceptable based on the type_filter
		// The type_filter is a bitmask where each bit represents a memory type that is suitable
		if (type_filter & (1 << i))
		{
			// Check if the memory type has all the desired property flags
			// properties is a bitmask of the required memory properties
			if ((mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				// Found a suitable memory type; return its index
				return i;
			}
		}
	}

	// If no suitable memory type was found, throw an exception
	throw std::runtime_error("Failed to find suitable memory type.");
}
/**
 * @brief Initializes per frame data.
 * @param per_frame The data of a frame.
 */
void HelloRayTracing::init_per_frame(PerFrame &per_frame)
{
	VkFenceCreateInfo info{
	    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
	    .flags = VK_FENCE_CREATE_SIGNALED_BIT};
	VK_CHECK(vkCreateFence(context.device, &info, nullptr, &per_frame.queue_submit_fence));

	VkCommandPoolCreateInfo cmd_pool_info{
	    .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	    .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
	    .queueFamilyIndex = static_cast<uint32_t>(context.graphics_queue_index)};
	VK_CHECK(vkCreateCommandPool(context.device, &cmd_pool_info, nullptr, &per_frame.primary_command_pool));

	VkCommandBufferAllocateInfo cmd_buf_info{
	    .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	    .commandPool        = per_frame.primary_command_pool,
	    .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	    .commandBufferCount = 1};
	VK_CHECK(vkAllocateCommandBuffers(context.device, &cmd_buf_info, &per_frame.primary_command_buffer));
}

/**
 * @brief Tears down the frame data.
 * @param per_frame The data of a frame.
 */
void HelloRayTracing::teardown_per_frame(PerFrame &per_frame)
{
	if (per_frame.queue_submit_fence != VK_NULL_HANDLE)
	{
		vkDestroyFence(context.device, per_frame.queue_submit_fence, nullptr);

		per_frame.queue_submit_fence = VK_NULL_HANDLE;
	}

	if (per_frame.primary_command_buffer != VK_NULL_HANDLE)
	{
		vkFreeCommandBuffers(context.device, per_frame.primary_command_pool, 1, &per_frame.primary_command_buffer);

		per_frame.primary_command_buffer = VK_NULL_HANDLE;
	}

	if (per_frame.primary_command_pool != VK_NULL_HANDLE)
	{
		vkDestroyCommandPool(context.device, per_frame.primary_command_pool, nullptr);

		per_frame.primary_command_pool = VK_NULL_HANDLE;
	}

	if (per_frame.swapchain_acquire_semaphore != VK_NULL_HANDLE)
	{
		vkDestroySemaphore(context.device, per_frame.swapchain_acquire_semaphore, nullptr);

		per_frame.swapchain_acquire_semaphore = VK_NULL_HANDLE;
	}

	if (per_frame.swapchain_release_semaphore != VK_NULL_HANDLE)
	{
		vkDestroySemaphore(context.device, per_frame.swapchain_release_semaphore, nullptr);

		per_frame.swapchain_release_semaphore = VK_NULL_HANDLE;
	}
}

/**
 * @brief Initializes the Vulkan swapchain.
 */
void HelloRayTracing::init_swapchain()
{
	VkSurfaceCapabilitiesKHR surface_properties;
	VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context.gpu, context.surface, &surface_properties));

	VkSurfaceFormatKHR format = vkb::select_surface_format(context.gpu, context.surface);

	VkExtent2D swapchain_size;
	if (surface_properties.currentExtent.width == 0xFFFFFFFF)
	{
		swapchain_size.width  = context.swapchain_dimensions.width;
		swapchain_size.height = context.swapchain_dimensions.height;
	}
	else
	{
		swapchain_size = surface_properties.currentExtent;
	}

	// FIFO must be supported by all implementations.
	VkPresentModeKHR swapchain_present_mode = VK_PRESENT_MODE_FIFO_KHR;

	// Determine the number of VkImage's to use in the swapchain.
	// Ideally, we desire to own 1 image at a time, the rest of the images can
	// either be rendered to and/or being queued up for display.
	uint32_t desired_swapchain_images = surface_properties.minImageCount + 1;
	if ((surface_properties.maxImageCount > 0) && (desired_swapchain_images > surface_properties.maxImageCount))
	{
		// Application must settle for fewer images than desired.
		desired_swapchain_images = surface_properties.maxImageCount;
	}

	// Figure out a suitable surface transform.
	VkSurfaceTransformFlagBitsKHR pre_transform;
	if (surface_properties.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
	{
		pre_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	}
	else
	{
		pre_transform = surface_properties.currentTransform;
	}

	VkSwapchainKHR old_swapchain = context.swapchain;

	// one bitmask needs to be set according to the priority of presentation engine
	VkCompositeAlphaFlagBitsKHR composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
	{
		composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	}
	else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
	{
		composite = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
	}
	else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
	{
		composite = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
	}
	else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
	{
		composite = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
	}

	VkSwapchainCreateInfoKHR info{
	    .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
	    .surface          = context.surface,                            // The surface onto which images will be presented
	    .minImageCount    = desired_swapchain_images,                   // Minimum number of images in the swapchain (number of buffers)
	    .imageFormat      = format.format,                              // Format of the swapchain images (e.g., VK_FORMAT_B8G8R8A8_SRGB)
	    .imageColorSpace  = format.colorSpace,                          // Color space of the images (e.g., VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
	    .imageExtent      = swapchain_size,                             // Resolution of the swapchain images (width and height)
	    .imageArrayLayers = 1,                                          // Number of layers in each image (usually 1 unless stereoscopic)
	    .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,        // How the images will be used (as color attachments)
	    .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,                  // Access mode of the images (exclusive to one queue family)
	    .preTransform     = pre_transform,                              // Transform to apply to images (e.g., rotation)
	    .compositeAlpha   = composite,                                  // Alpha blending to apply (e.g., opaque, pre-multiplied)
	    .presentMode      = swapchain_present_mode,                     // Presentation mode (e.g., vsync settings)
	    .clipped          = true,                                       // Whether to clip obscured pixels (improves performance)
	    .oldSwapchain     = old_swapchain                               // Handle to the old swapchain, if replacing an existing one
	};

	VK_CHECK(vkCreateSwapchainKHR(context.device, &info, nullptr, &context.swapchain));

	if (old_swapchain != VK_NULL_HANDLE)
	{
		for (VkImageView image_view : context.swapchain_image_views)
		{
			vkDestroyImageView(context.device, image_view, nullptr);
		}

		for (auto &per_frame : context.per_frame)
		{
			teardown_per_frame(per_frame);
		}

		context.swapchain_image_views.clear();

		vkDestroySwapchainKHR(context.device, old_swapchain, nullptr);
	}

	context.swapchain_dimensions = {swapchain_size.width, swapchain_size.height, format.format};

	uint32_t image_count;
	VK_CHECK(vkGetSwapchainImagesKHR(context.device, context.swapchain, &image_count, nullptr));

	/// The swapchain images.
	std::vector<VkImage> swapchain_images(image_count);
	VK_CHECK(vkGetSwapchainImagesKHR(context.device, context.swapchain, &image_count, swapchain_images.data()));

	// Store swapchain images
	context.swapchain_images = swapchain_images;

	// Initialize per-frame resources.
	// Every swapchain image has its own command pool and fence manager.
	// This makes it very easy to keep track of when we can reset command buffers and such.
	context.per_frame.clear();
	context.per_frame.resize(image_count);

	for (size_t i = 0; i < image_count; i++)
	{
		init_per_frame(context.per_frame[i]);
	}

	for (size_t i = 0; i < image_count; i++)
	{
		// Create an image view which we can render into.
		VkImageViewCreateInfo view_info{
		    .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		    .pNext            = nullptr,
		    .flags            = 0,
		    .image            = swapchain_images[i],
		    .viewType         = VK_IMAGE_VIEW_TYPE_2D,
		    .format           = context.swapchain_dimensions.format,
		    .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};

		VkImageView image_view;
		VK_CHECK(vkCreateImageView(context.device, &view_info, nullptr, &image_view));

		context.swapchain_image_views.push_back(image_view);
	}
}
/**
 * @brief Helper function to load a shader module.
 * @param path The path for the shader (relative to the assets directory).
 * @param shader_stage The shader stage flag specifying the type of shader (e.g., VK_SHADER_STAGE_VERTEX_BIT).
 * @returns A VkShaderModule handle. Aborts execution if shader creation fails.
 */
VkShaderModule HelloRayTracing::load_shader_module(const std::string &path, VkShaderStageFlagBits shader_stage)
{
	auto spirv = vkb::fs::read_shader_binary_u32(path);

	VkShaderModuleCreateInfo module_info{
	    .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
	    .codeSize = spirv.size() * sizeof(uint32_t),
	    .pCode    = spirv.data()};

	VkShaderModule shader_module;
	VK_CHECK(vkCreateShaderModule(context.device, &module_info, nullptr, &shader_module));

	return shader_module;
}

/**
 * @brief Initializes the Vulkan pipeline.
 */
void HelloRayTracing::init_pipeline()
{
	// Create a blank pipeline layout.
	// We are not binding any resources to the pipeline in this first sample.
	VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	VK_CHECK(vkCreatePipelineLayout(context.device, &layout_info, nullptr, &context.pipeline_layout));

	// Define the vertex input binding description
	VkVertexInputBindingDescription binding_description{
	    .binding   = 0,
	    .stride    = sizeof(Vertex),
	    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};

	// Define the vertex input attribute descriptions
	std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions = {{
	    {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, position)},        // position
	    {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color)}         // color
	}};

	// Create the vertex input state
	VkPipelineVertexInputStateCreateInfo vertex_input{.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	                                                  .vertexBindingDescriptionCount   = 1,
	                                                  .pVertexBindingDescriptions      = &binding_description,
	                                                  .vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descriptions.size()),
	                                                  .pVertexAttributeDescriptions    = attribute_descriptions.data()};

	// Specify we will use triangle lists to draw geometry.
	VkPipelineInputAssemblyStateCreateInfo input_assembly{
	    .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
	    .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	    .primitiveRestartEnable = VK_FALSE};

	// Specify rasterization state.
	VkPipelineRasterizationStateCreateInfo raster{
	    .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
	    .depthClampEnable        = VK_FALSE,
	    .rasterizerDiscardEnable = VK_FALSE,
	    .polygonMode             = VK_POLYGON_MODE_FILL,
	    .depthBiasEnable         = VK_FALSE,
	    .lineWidth               = 1.0f};

	// Specify that these states will be dynamic, i.e. not part of pipeline state object.
	std::vector<VkDynamicState> dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR,
	    VK_DYNAMIC_STATE_CULL_MODE,
	    VK_DYNAMIC_STATE_FRONT_FACE,
	    VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY};

	// Our attachment will write to all color channels, but no blending is enabled.
	VkPipelineColorBlendAttachmentState blend_attachment{
	    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};

	VkPipelineColorBlendStateCreateInfo blend{
	    .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
	    .attachmentCount = 1,
	    .pAttachments    = &blend_attachment};

	// We will have one viewport and scissor box.
	VkPipelineViewportStateCreateInfo viewport{
	    .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
	    .viewportCount = 1,
	    .scissorCount  = 1};

	// Disable all depth testing.
	VkPipelineDepthStencilStateCreateInfo depth_stencil{
	    .sType          = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
	    .depthCompareOp = VK_COMPARE_OP_ALWAYS};

	// No multisampling.
	VkPipelineMultisampleStateCreateInfo multisample{
	    .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
	    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};

	VkPipelineDynamicStateCreateInfo dynamic_state_info{
	    .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
	    .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
	    .pDynamicStates    = dynamic_states.data()};

	// Load our SPIR-V shaders.

	// Samples support different shading languages, all of which are offline compiled to SPIR-V, the shader format that Vulkan uses.
	// The shading language to load for can be selected via command line
	std::string shader_folder{""};
	switch (get_shading_language())
	{
		case vkb::ShadingLanguage::HLSL:
			shader_folder = "hlsl";
			break;
		case vkb::ShadingLanguage::SLANG:
			shader_folder = "slang";
			break;
		default:
			shader_folder = "glsl";
	}

	std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages = {{
	    {.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	     .stage  = VK_SHADER_STAGE_VERTEX_BIT,
	     .module = load_shader_module("hello_ray_tracing/" + shader_folder + "/triangle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
	     .pName  = "main"},        // Vertex shader stage
	    {
	        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	        .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
	        .module = load_shader_module("hello_ray_tracing/" + shader_folder + "/triangle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
	        .pName  = "main"}        // Fragment shader stage
	}};

	// Pipeline rendering info (for dynamic rendering).
	VkPipelineRenderingCreateInfo pipeline_rendering_info{
	    .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
	    .colorAttachmentCount    = 1,
	    .pColorAttachmentFormats = &context.swapchain_dimensions.format};

	// Create the graphics pipeline.
	VkGraphicsPipelineCreateInfo pipe{
	    .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
	    .pNext               = &pipeline_rendering_info,
	    .stageCount          = vkb::to_u32(shader_stages.size()),
	    .pStages             = shader_stages.data(),
	    .pVertexInputState   = &vertex_input,
	    .pInputAssemblyState = &input_assembly,
	    .pViewportState      = &viewport,
	    .pRasterizationState = &raster,
	    .pMultisampleState   = &multisample,
	    .pDepthStencilState  = &depth_stencil,
	    .pColorBlendState    = &blend,
	    .pDynamicState       = &dynamic_state_info,
	    .layout              = context.pipeline_layout,        // We need to specify the pipeline layout description up front as well.
	    .renderPass          = VK_NULL_HANDLE,                 // Since we are using dynamic rendering this will set as null
	    .subpass             = 0,
	};

	VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipe, nullptr, &context.pipeline));

	// Pipeline is baked, we can delete the shader modules now.
	vkDestroyShaderModule(context.device, shader_stages[0].module, nullptr);
	vkDestroyShaderModule(context.device, shader_stages[1].module, nullptr);
}

/**
 * @brief Acquires an image from the swapchain.
 * @param[out] image The swapchain index for the acquired image.
 * @returns Vulkan result code
 */
VkResult HelloRayTracing::acquire_next_swapchain_image(uint32_t *image)
{
	VkSemaphore acquire_semaphore;
	if (context.recycled_semaphores.empty())
	{
		VkSemaphoreCreateInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
		VK_CHECK(vkCreateSemaphore(context.device, &info, nullptr, &acquire_semaphore));
	}
	else
	{
		acquire_semaphore = context.recycled_semaphores.back();
		context.recycled_semaphores.pop_back();
	}

	VkResult res = vkAcquireNextImageKHR(context.device, context.swapchain, UINT64_MAX, acquire_semaphore, VK_NULL_HANDLE, image);

	if (res != VK_SUCCESS)
	{
		context.recycled_semaphores.push_back(acquire_semaphore);
		return res;
	}

	// If we have outstanding fences for this swapchain image, wait for them to complete first.
	// After begin frame returns, it is safe to reuse or delete resources which
	// were used previously.
	//
	// We wait for fences which completes N frames earlier, so we do not stall,
	// waiting for all GPU work to complete before this returns.
	// Normally, this doesn't really block at all,
	// since we're waiting for old frames to have been completed, but just in case.
	if (context.per_frame[*image].queue_submit_fence != VK_NULL_HANDLE)
	{
		vkWaitForFences(context.device, 1, &context.per_frame[*image].queue_submit_fence, true, UINT64_MAX);
		vkResetFences(context.device, 1, &context.per_frame[*image].queue_submit_fence);
	}

	if (context.per_frame[*image].primary_command_pool != VK_NULL_HANDLE)
	{
		vkResetCommandPool(context.device, context.per_frame[*image].primary_command_pool, 0);
	}

	// Recycle the old semaphore back into the semaphore manager.
	VkSemaphore old_semaphore = context.per_frame[*image].swapchain_acquire_semaphore;

	if (old_semaphore != VK_NULL_HANDLE)
	{
		context.recycled_semaphores.push_back(old_semaphore);
	}

	context.per_frame[*image].swapchain_acquire_semaphore = acquire_semaphore;

	return VK_SUCCESS;
}

/**
 * @brief Renders a triangle to the specified swapchain image.
 * @param swapchain_index The swapchain index for the image being rendered.
 */
void HelloRayTracing::render_triangle(uint32_t swapchain_index)
{
	// Allocate or re-use a primary command buffer.
	VkCommandBuffer cmd = context.per_frame[swapchain_index].primary_command_buffer;

	// We will only submit this once before it's recycled.
	VkCommandBufferBeginInfo begin_info{
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

	// Begin command recording
	VK_CHECK(vkBeginCommandBuffer(cmd, &begin_info));

	// Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
	transition_image_layout(
	    cmd,
	    context.swapchain_images[swapchain_index],
	    VK_IMAGE_LAYOUT_UNDEFINED,
	    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	    0,                                                     // srcAccessMask (no need to wait for previous operations)
	    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,                // dstAccessMask
	    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,                   // srcStage
	    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT        // dstStage
	);
	// Set clear color values.
	VkClearValue clear_value{
	    .color = {{0.1f, 0.65f, 0.1f, 1.0f}}};

	// Set up the rendering attachment info
	VkRenderingAttachmentInfo color_attachment{
	    .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
	    .imageView   = context.swapchain_image_views[swapchain_index],
	    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	    .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
	    .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
	    .clearValue  = clear_value};

	// Begin rendering
	VkRenderingInfo rendering_info{
	    .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
	    .renderArea           = {                         // Initialize the nested `VkRect2D` structure
	                             .offset = {0, 0},        // Initialize the `VkOffset2D` inside `renderArea`
	                             .extent = {              // Initialize the `VkExtent2D` inside `renderArea`
	                                        .width  = context.swapchain_dimensions.width,
	                                        .height = context.swapchain_dimensions.height}},
	              .layerCount = 1,
	    .colorAttachmentCount = 1,
	    .pColorAttachments    = &color_attachment};

	vkCmdBeginRendering(cmd, &rendering_info);

	// Bind the graphics pipeline.
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, context.pipeline);

	// Set dynamic states

	// Set viewport dynamically
	VkViewport vp{
	    .width    = static_cast<float>(context.swapchain_dimensions.width),
	    .height   = static_cast<float>(context.swapchain_dimensions.height),
	    .minDepth = 0.0f,
	    .maxDepth = 1.0f};

	vkCmdSetViewport(cmd, 0, 1, &vp);

	// Set scissor dynamically
	VkRect2D scissor{
	    .extent = {
	        .width  = context.swapchain_dimensions.width,
	        .height = context.swapchain_dimensions.height}};

	vkCmdSetScissor(cmd, 0, 1, &scissor);

	// Since we declared VK_DYNAMIC_STATE_CULL_MODE as dynamic in the pipeline,
	// we need to set the cull mode here. VK_CULL_MODE_NONE disables face culling,
	// meaning both front and back faces will be rendered.
	vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);

	// Since we declared VK_DYNAMIC_STATE_FRONT_FACE as dynamic,
	// we need to specify the winding order considered as the front face.
	// VK_FRONT_FACE_CLOCKWISE indicates that vertices defined in clockwise order
	// are considered front-facing.
	vkCmdSetFrontFace(cmd, VK_FRONT_FACE_CLOCKWISE);

	// Since we declared VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY as dynamic,
	// we need to set the primitive topology here. VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
	// tells Vulkan that the input vertex data should be interpreted as a list of triangles.
	vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	// Bind the vertex buffer
	VkDeviceSize offset = {0};
	vkCmdBindVertexBuffers(cmd, 0, 1, &context.vertex_buffer, &offset);

	// Draw three vertices with one instance.
	vkCmdDraw(cmd, vertices.size(), 1, 0, 0);

	// Complete rendering.
	vkCmdEndRendering(cmd);

	// After rendering , transition the swapchain image to PRESENT_SRC
	transition_image_layout(
	    cmd,
	    context.swapchain_images[swapchain_index],
	    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,                 // srcAccessMask
	    0,                                                      // dstAccessMask
	    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,        // srcStage
	    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT                  // dstStage
	);

	// Complete the command buffer.
	VK_CHECK(vkEndCommandBuffer(cmd));

	// Submit it to the queue with a release semaphore.
	if (context.per_frame[swapchain_index].swapchain_release_semaphore == VK_NULL_HANDLE)
	{
		VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
		VK_CHECK(vkCreateSemaphore(context.device, &semaphore_info, nullptr,
		                           &context.per_frame[swapchain_index].swapchain_release_semaphore));
	}

	// Using TOP_OF_PIPE here to ensure that the command buffer does not begin executing any pipeline stages
	// (including the layout transition) until the swapchain image is actually acquired (signaled by the semaphore).
	// This prevents the GPU from starting operations too early and guarantees that the image is ready
	// before any rendering commands run.
	VkPipelineStageFlags wait_stage{VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT};

	VkSubmitInfo info{
	    .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
	    .waitSemaphoreCount   = 1,
	    .pWaitSemaphores      = &context.per_frame[swapchain_index].swapchain_acquire_semaphore,
	    .pWaitDstStageMask    = &wait_stage,
	    .commandBufferCount   = 1,
	    .pCommandBuffers      = &cmd,
	    .signalSemaphoreCount = 1,
	    .pSignalSemaphores    = &context.per_frame[swapchain_index].swapchain_release_semaphore};

	// Submit command buffer to graphics queue
	VK_CHECK(vkQueueSubmit(context.queue, 1, &info, context.per_frame[swapchain_index].queue_submit_fence));
}

/**
 * @brief Presents an image to the swapchain.
 * @param index The swapchain index previously obtained from @ref acquire_next_swapchain_image.
 * @returns Vulkan result code
 */
VkResult HelloRayTracing::present_image(uint32_t index)
{
	VkPresentInfoKHR present{
	    .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
	    .waitSemaphoreCount = 1,
	    .pWaitSemaphores    = &context.per_frame[index].swapchain_release_semaphore,
	    .swapchainCount     = 1,
	    .pSwapchains        = &context.swapchain,
	    .pImageIndices      = &index,
	};

	// Present swapchain image
	return vkQueuePresentKHR(context.queue, &present);
}

/**
 * @brief Transitions an image layout in a Vulkan command buffer.
 * @param cmd The command buffer to record the barrier into.
 * @param image The Vulkan image to transition.
 * @param oldLayout The current layout of the image.
 * @param newLayout The desired new layout of the image.
 * @param srcAccessMask The source access mask, specifying which access types are being transitioned from.
 * @param dstAccessMask The destination access mask, specifying which access types are being transitioned to.
 * @param srcStage The pipeline stage that must happen before the transition.
 * @param dstStage The pipeline stage that must happen after the transition.
 */
void HelloRayTracing::transition_image_layout(
    VkCommandBuffer       cmd,
    VkImage               image,
    VkImageLayout         oldLayout,
    VkImageLayout         newLayout,
    VkAccessFlags2        srcAccessMask,
    VkAccessFlags2        dstAccessMask,
    VkPipelineStageFlags2 srcStage,
    VkPipelineStageFlags2 dstStage)
{
	// Initialize the VkImageMemoryBarrier2 structure
	VkImageMemoryBarrier2 image_barrier{
	    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,

	    // Specify the pipeline stages and access masks for the barrier
	    .srcStageMask  = srcStage,             // Source pipeline stage mask
	    .srcAccessMask = srcAccessMask,        // Source access mask
	    .dstStageMask  = dstStage,             // Destination pipeline stage mask
	    .dstAccessMask = dstAccessMask,        // Destination access mask

	    // Specify the old and new layouts of the image
	    .oldLayout = oldLayout,        // Current layout of the image
	    .newLayout = newLayout,        // Target layout of the image

	    // We are not changing the ownership between queues
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

	    // Specify the image to be affected by this barrier
	    .image = image,

	    // Define the subresource range (which parts of the image are affected)
	    .subresourceRange = {
	        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,        // Affects the color aspect of the image
	        .baseMipLevel   = 0,                                // Start at mip level 0
	        .levelCount     = 1,                                // Number of mip levels affected
	        .baseArrayLayer = 0,                                // Start at array layer 0
	        .layerCount     = 1                                 // Number of array layers affected
	    }};

	// Initialize the VkDependencyInfo structure
	VkDependencyInfo dependency_info{
	    .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
	    .dependencyFlags         = 0,                    // No special dependency flags
	    .imageMemoryBarrierCount = 1,                    // Number of image memory barriers
	    .pImageMemoryBarriers    = &image_barrier        // Pointer to the image memory barrier(s)
	};

	// Record the pipeline barrier into the command buffer
	vkCmdPipelineBarrier2(cmd, &dependency_info);
}

HelloRayTracing::~HelloRayTracing()
{
	// Don't release anything until the GPU is completely idle.
	if (context.device != VK_NULL_HANDLE)
	{
		vkDeviceWaitIdle(context.device);
	}

	for (auto &per_frame : context.per_frame)
	{
		teardown_per_frame(per_frame);
	}

	context.per_frame.clear();

	for (auto semaphore : context.recycled_semaphores)
	{
		vkDestroySemaphore(context.device, semaphore, nullptr);
	}

	if (context.pipeline != VK_NULL_HANDLE)
	{
		vkDestroyPipeline(context.device, context.pipeline, nullptr);
	}

	if (context.pipeline_layout != VK_NULL_HANDLE)
	{
		vkDestroyPipelineLayout(context.device, context.pipeline_layout, nullptr);
	}

	for (VkImageView image_view : context.swapchain_image_views)
	{
		vkDestroyImageView(context.device, image_view, nullptr);
	}

	if (context.swapchain != VK_NULL_HANDLE)
	{
		vkDestroySwapchainKHR(context.device, context.swapchain, nullptr);
		context.swapchain = VK_NULL_HANDLE;
	}

	if (context.surface != VK_NULL_HANDLE)
	{
		vkDestroySurfaceKHR(context.instance, context.surface, nullptr);
		context.surface = VK_NULL_HANDLE;
	}

	if (context.vertex_buffer != VK_NULL_HANDLE)
	{
		vkDestroyBuffer(context.device, context.vertex_buffer, nullptr);
		context.vertex_buffer = VK_NULL_HANDLE;
	}

	if (context.vertex_buffer_memory != VK_NULL_HANDLE)
	{
		vkFreeMemory(context.device, context.vertex_buffer_memory, nullptr);
		context.vertex_buffer_memory = VK_NULL_HANDLE;
	}

	if (context.device != VK_NULL_HANDLE)
	{
		vkDestroyDevice(context.device, nullptr);
		context.device = VK_NULL_HANDLE;
	}

	if (context.debug_callback != VK_NULL_HANDLE)
	{
		vkDestroyDebugUtilsMessengerEXT(context.instance, context.debug_callback, nullptr);
		context.debug_callback = VK_NULL_HANDLE;
	}

	vk_instance.reset();
}

bool HelloRayTracing::prepare(const vkb::ApplicationOptions &options)
{
	assert(options.window != nullptr);

	init_instance();

	vk_instance = std::make_unique<vkb::core::InstanceC>(context.instance);

	context.surface                     = options.window->create_surface(*vk_instance);
	auto &extent                        = options.window->get_extent();
	context.swapchain_dimensions.width  = extent.width;
	context.swapchain_dimensions.height = extent.height;

	if (!context.surface)
	{
		throw std::runtime_error("Failed to create window surface.");
	}

	init_device();

	init_vertex_buffer();

	init_swapchain();

	// Create the necessary objects for rendering.
	init_pipeline();

	return true;
}

void HelloRayTracing::update(float delta_time)
{
	uint32_t index;

	auto res = acquire_next_swapchain_image(&index);

	// Handle outdated error in acquire.
	if (res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR)
	{
		if (!resize(context.swapchain_dimensions.width, context.swapchain_dimensions.height))
		{
			LOGI("Resize failed");
		}
		res = acquire_next_swapchain_image(&index);
	}

	if (res != VK_SUCCESS)
	{
		vkQueueWaitIdle(context.queue);
		return;
	}

	render_triangle(index);
	res = present_image(index);

	// Handle Outdated error in present.
	if (res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR)
	{
		if (!resize(context.swapchain_dimensions.width, context.swapchain_dimensions.height))
		{
			LOGI("Resize failed");
		}
	}
	else if (res != VK_SUCCESS)
	{
		LOGE("Failed to present swapchain image.");
	}
}

bool HelloRayTracing::resize(const uint32_t, const uint32_t)
{
	if (context.device == VK_NULL_HANDLE)
	{
		return false;
	}

	VkSurfaceCapabilitiesKHR surface_properties;
	VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context.gpu, context.surface, &surface_properties));

	// Only rebuild the swapchain if the dimensions have changed
	if (surface_properties.currentExtent.width == context.swapchain_dimensions.width &&
	    surface_properties.currentExtent.height == context.swapchain_dimensions.height)
	{
		return false;
	}

	vkDeviceWaitIdle(context.device);

	init_swapchain();
	return true;
}

std::unique_ptr<vkb::Application> create_hello_ray_tracing()
{
	return std::make_unique<HelloRayTracing>();
}

//////////////////////
// BEGIN TUTORIAL PORT
//////////////////////

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <limits>
#include <array>
#include <chrono>

#include <vulkan/vulkan_raii.hpp>

/* // OLD TUTORIAL CODE.
#ifdef __INTELLISENSE__
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
*/

#include <vulkan/vk_platform.h>

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
//#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

/* // OLD TUTORIAL CODE. No longer needed?
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
*/

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#ifndef LAB_TASK_LEVEL
#define LAB_TASK_LEVEL 1
#endif

#define LAB_TASK_AS_BUILD_AND_BIND 4
#define LAB_TASK_AS_ANIMATION 6
#define LAB_TASK_AS_OPAQUE_FLAG 7
#define LAB_TASK_INSTANCE_LUT 9
#define LAB_TASK_REFLECTIONS 11

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint64_t FenceTimeout = 100000000;
const std::string MODEL_PATH = "models/plant_on_table.obj";
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector validationLayers = {
        "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
    glm::vec3 normal;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions() {
        return {
                vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) ),
                vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) ),
                vk::VertexInputAttributeDescription( 2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord) ),
                vk::VertexInputAttributeDescription( 3, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal) )
        };
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord && normal == other.normal;
    }
};

template<> struct std::hash<Vertex> {
    size_t operator()(Vertex const& vertex) const noexcept {
        auto h = std::hash<glm::vec3>()(vertex.pos) ^ (std::hash<glm::vec3>()(vertex.color) << 1);
        h = (h >> 1) ^ (std::hash<glm::vec2>()(vertex.texCoord) << 1);
        h = (h >> 1) ^ (std::hash<glm::vec3>()(vertex.normal) << 1);
        return h;
    }
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 cameraPos;
};

struct PushConstant {
    uint32_t materialIndex;
#if LAB_TASK_LEVEL >= LAB_TASK_REFLECTIONS
    // TASK11
    uint32_t reflective;
#endif // LAB_TASK_LEVEL >= LAB_TASK_REFLECTIONS
};

#if false

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    vk::raii::Context  context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;

    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;

    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;

    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::DescriptorSetLayout descriptorSetLayoutGlobal = nullptr;
    vk::raii::DescriptorSetLayout descriptorSetLayoutMaterial = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    std::vector<vk::raii::Image> textureImages;
    std::vector<vk::raii::DeviceMemory> textureImageMemories;
    std::vector<vk::raii::ImageView> textureImageViews;
    vk::raii::Sampler textureSampler = nullptr;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;
    vk::raii::Buffer uvBuffer = nullptr;
    vk::raii::DeviceMemory uvBufferMemory = nullptr;

    std::vector<vk::raii::Buffer> blasBuffers;
    std::vector<vk::raii::DeviceMemory> blasMemories;
    std::vector<vk::raii::AccelerationStructureKHR> blasHandles;

    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    vk::raii::Buffer instanceBuffer = nullptr;
    vk::raii::DeviceMemory instanceMemory = nullptr;

    vk::raii::Buffer tlasBuffer = nullptr;
    vk::raii::DeviceMemory tlasMemory = nullptr;
    vk::raii::Buffer tlasScratchBuffer = nullptr;
    vk::raii::DeviceMemory tlasScratchMemory = nullptr;
    vk::raii::AccelerationStructureKHR tlas = nullptr;

    struct InstanceLUT {
        uint32_t materialID;
        uint32_t indexBufferOffset;
    };
    std::vector<InstanceLUT> instanceLUTs;
    vk::raii::Buffer instanceLUTBuffer = nullptr;
    vk::raii::DeviceMemory instanceLUTBufferMemory = nullptr;

    UniformBufferObject ubo{};

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    struct SubMesh {
        uint32_t indexOffset;
        uint32_t indexCount;
        int materialID;
        uint32_t firstVertex;
        uint32_t maxVertex;
        bool alphaCut;
        bool reflective;
    };
    std::vector<SubMesh> submeshes;
    std::vector<tinyobj::material_t> materials;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> globalDescriptorSets;
    std::vector<vk::raii::DescriptorSet> materialDescriptorSets;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    uint32_t graphicsIndex = 0;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphore;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphore;
    std::vector<vk::raii::Fence> inFlightFences;
    uint32_t semaphoreIndex = 0;
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    std::vector<const char*> requiredDeviceExtension = {
            vk::KHRSwapchainExtensionName,
            vk::KHRSpirv14ExtensionName,
            vk::KHRSynchronization2ExtensionName,
            vk::KHRCreateRenderpass2ExtensionName,
            vk::KHRAccelerationStructureExtensionName,
            vk::KHRBufferDeviceAddressExtensionName,
            vk::KHRDeferredHostOperationsExtensionName,
            vk::KHRRayQueryExtensionName
    };

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = static_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createCommandPool();
        loadModel();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createDepthResources();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUVBuffer();
        createAccelerationStructures();
        createInstanceLUTBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        device.waitIdle();
    }

    void cleanupSwapChain() {
        swapChainImageViews.clear();
        swapChain = nullptr;
    }

    void cleanup() const {
        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createDepthResources();
    }

    void createInstance() {
        constexpr vk::ApplicationInfo appInfo{ .pApplicationName   = "Hello Triangle",
                .applicationVersion = VK_MAKE_VERSION( 1, 0, 0 ),
                .pEngineName        = "No Engine",
                .engineVersion      = VK_MAKE_VERSION( 1, 0, 0 ),
                .apiVersion         = vk::ApiVersion14 };

        // Get the required layers
        std::vector<char const*> requiredLayers;
        if (enableValidationLayers) {
            requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        // Check if the required layers are supported by the Vulkan implementation.
        auto layerProperties = context.enumerateInstanceLayerProperties();
        for (auto const& requiredLayer : requiredLayers)
        {
            if (std::ranges::none_of(layerProperties,
                                     [requiredLayer](auto const& layerProperty)
                                     { return strcmp(layerProperty.layerName, requiredLayer) == 0; }))
            {
                throw std::runtime_error("Required layer not supported: " + std::string(requiredLayer));
            }
        }

        // Get the required extensions.
        auto requiredExtensions = getRequiredExtensions();

        // Check if the required extensions are supported by the Vulkan implementation.
        auto extensionProperties = context.enumerateInstanceExtensionProperties();
        for (auto const& requiredExtension : requiredExtensions)
        {
            if (std::ranges::none_of(extensionProperties,
                                     [requiredExtension](auto const& extensionProperty)
                                     { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; }))
            {
                throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
            }
        }

        vk::InstanceCreateInfo createInfo{
                .pApplicationInfo        = &appInfo,
                .enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
                .ppEnabledLayerNames     = requiredLayers.data(),
                .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
                .ppEnabledExtensionNames = requiredExtensions.data() };
        instance = vk::raii::Instance(context, createInfo);
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags( vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError );
        vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags( vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation );
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
                .messageSeverity = severityFlags,
                .messageType = messageTypeFlags,
                .pfnUserCallback = &debugCallback
        };
        debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }

    void createSurface() {
        VkSurfaceKHR       _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void pickPhysicalDevice() {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        const auto                            devIter = std::ranges::find_if(
                devices,
                [&]( auto const & device )
                {
                    // Check if the device supports the Vulkan 1.3 API version
                    bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

                    // Check if any of the queue families support graphics operations
                    auto queueFamilies = device.getQueueFamilyProperties();
                    bool supportsGraphics =
                            std::ranges::any_of( queueFamilies, []( auto const & qfp ) { return !!( qfp.queueFlags & vk::QueueFlagBits::eGraphics ); } );

                    // Check if all required device extensions are available
                    auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
                    bool supportsAllRequiredExtensions =
                            std::ranges::all_of( requiredDeviceExtension,
                                                 [&availableDeviceExtensions]( auto const & requiredDeviceExtension )
                                                 {
                                                     return std::ranges::any_of( availableDeviceExtensions,
                                                                                 [requiredDeviceExtension]( auto const & availableDeviceExtension )
                                                                                 { return strcmp( availableDeviceExtension.extensionName, requiredDeviceExtension ) == 0; } );
                                                 } );

                    auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2,
                            vk::PhysicalDeviceVulkan12Features,
                            vk::PhysicalDeviceVulkan13Features,
                            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                            vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                            vk::PhysicalDeviceRayQueryFeaturesKHR>();
                    bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                                                    features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                                    features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
                                                    features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingSampledImageUpdateAfterBind &&
                                                    features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingPartiallyBound &&
                                                    features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingVariableDescriptorCount &&
                                                    features.template get<vk::PhysicalDeviceVulkan12Features>().runtimeDescriptorArray &&
                                                    features.template get<vk::PhysicalDeviceVulkan12Features>().shaderSampledImageArrayNonUniformIndexing &&
                                                    features.template get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress &&
                                                    features.template get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure &&
                                                    features.template get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery;

                    return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
                } );
        if ( devIter != devices.end() )
        {
            physicalDevice = *devIter;
        }
        else
        {
            throw std::runtime_error( "failed to find a suitable GPU!" );
        }
    }

    void createLogicalDevice() {
        // find the index of the first queue family that supports graphics
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports graphics
        auto graphicsQueueFamilyProperty = std::ranges::find_if( queueFamilyProperties, []( auto const & qfp )
        { return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0); } );

        graphicsIndex = static_cast<uint32_t>( std::distance( queueFamilyProperties.begin(), graphicsQueueFamilyProperty ) );

        // determine a queueFamilyIndex that supports present
        // first check if the graphicsIndex is good enough
        auto presentIndex = physicalDevice.getSurfaceSupportKHR( graphicsIndex, *surface )
                            ? graphicsIndex
                            : ~0;
        if ( presentIndex == queueFamilyProperties.size() )
        {
            // the graphicsIndex doesn't support present -> look for another family index that supports both
            // graphics and present
            for ( size_t i = 0; i < queueFamilyProperties.size(); i++ )
            {
                if ( ( queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics ) &&
                     physicalDevice.getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
                {
                    graphicsIndex = static_cast<uint32_t>( i );
                    presentIndex  = graphicsIndex;
                    break;
                }
            }
            if ( presentIndex == queueFamilyProperties.size() )
            {
                // there's nothing like a single family index that supports both graphics and present -> look for another
                // family index that supports present
                for ( size_t i = 0; i < queueFamilyProperties.size(); i++ )
                {
                    if ( physicalDevice.getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
                    {
                        presentIndex = static_cast<uint32_t>( i );
                        break;
                    }
                }
            }
        }
        if ( ( graphicsIndex == queueFamilyProperties.size() ) || ( presentIndex == queueFamilyProperties.size() ) )
        {
            throw std::runtime_error( "Could not find a queue for graphics or present -> terminating" );
        }

        // query for Vulkan 1.3 features
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
                vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceRayQueryFeaturesKHR> featureChain = {
                {.features = {.samplerAnisotropy = true } },                       // vk::PhysicalDeviceFeatures2
                {.shaderSampledImageArrayNonUniformIndexing = true, .descriptorBindingSampledImageUpdateAfterBind = true,
                        .descriptorBindingPartiallyBound = true, .descriptorBindingVariableDescriptorCount = true,
                        .runtimeDescriptorArray = true, .bufferDeviceAddress = true },    // vk::PhysicalDeviceVulkan12Features
                {.synchronization2 = true, .dynamicRendering = true },             // vk::PhysicalDeviceVulkan13Features
                {.extendedDynamicState = true },                                   // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
                {.accelerationStructure = true },                                  // vk::PhysicalDeviceAccelerationStructureFeaturesKHR
                {.rayQuery = true }                                                // vk::PhysicalDeviceRayQueryFeaturesKHR
        };

        // create a Device
        float                     queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ .queueFamilyIndex = graphicsIndex, .queueCount = 1, .pQueuePriorities = &queuePriority };
        vk::DeviceCreateInfo      deviceCreateInfo{ .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
                .queueCreateInfoCount = 1,
                .pQueueCreateInfos = &deviceQueueCreateInfo,
                .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
                .ppEnabledExtensionNames = requiredDeviceExtension.data() };

        device = vk::raii::Device( physicalDevice, deviceCreateInfo );
        graphicsQueue = vk::raii::Queue( device, graphicsIndex, 0 );
        presentQueue = vk::raii::Queue( device, presentIndex, 0 );
    }

    void createSwapChain() {
        auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
        swapChainImageFormat = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR( surface ));
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);
        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
        minImageCount = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount : minImageCount;
        vk::SwapchainCreateInfoKHR swapChainCreateInfo{
                .surface = surface, .minImageCount = minImageCount,
                .imageFormat = swapChainImageFormat, .imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
                .imageExtent = swapChainExtent, .imageArrayLayers =1,
                .imageUsage = vk::ImageUsageFlagBits::eColorAttachment, .imageSharingMode = vk::SharingMode::eExclusive,
                .preTransform = surfaceCapabilities.currentTransform, .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                .presentMode = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface)),
                .clipped = true };

        swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
        swapChainImages = swapChain.getImages();
    }

    void createImageViews() {
        vk::ImageViewCreateInfo imageViewCreateInfo{
                .viewType = vk::ImageViewType::e2D,
                .format = swapChainImageFormat,
                .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
        };
        for ( auto image : swapChainImages )
        {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back( device, imageViewCreateInfo );
        }
    }

    void createDescriptorSetLayout() {
        // Use descriptor set 0 for global data
        // TASK04: The acceleration structure uses binding 1
        std::array global_bindings = {
                vk::DescriptorSetLayoutBinding( 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),
                vk::DescriptorSetLayoutBinding( 1, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
                vk::DescriptorSetLayoutBinding( 2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
                vk::DescriptorSetLayoutBinding( 3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
                vk::DescriptorSetLayoutBinding( 4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
        };

        vk::DescriptorSetLayoutCreateInfo globalLayoutInfo{ .bindingCount = static_cast<uint32_t>(global_bindings.size()), .pBindings = global_bindings.data() };

        descriptorSetLayoutGlobal = vk::raii::DescriptorSetLayout(device, globalLayoutInfo);

        // Use descriptor set 1 for bindless material data
        uint32_t textureCount = static_cast<uint32_t>(textureImageViews.size());

        std::array material_bindings = {
                vk::DescriptorSetLayoutBinding( 0, vk::DescriptorType::eSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
                vk::DescriptorSetLayoutBinding( 1, vk::DescriptorType::eSampledImage, static_cast<uint32_t>(textureCount), vk::ShaderStageFlagBits::eFragment, nullptr)
        };

        std::vector<vk::DescriptorBindingFlags> bindingFlags = {
                vk::DescriptorBindingFlagBits::eUpdateAfterBind,
                vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eVariableDescriptorCount | vk::DescriptorBindingFlagBits::eUpdateAfterBind
        };

        vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo{
                .bindingCount = static_cast<uint32_t>(bindingFlags.size()),
                .pBindingFlags = bindingFlags.data()
        };

        vk::DescriptorSetLayoutCreateInfo materialLayoutInfo{
                .pNext = &flagsCreateInfo,
                .flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
                .bindingCount = static_cast<uint32_t>(material_bindings.size()),
                .pBindings = material_bindings.data(),
        };

        descriptorSetLayoutMaterial = vk::raii::DescriptorSetLayout(device, materialLayoutInfo);
    }

    void createGraphicsPipeline() {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule,  .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &bindingDescription,
                .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
                .pVertexAttributeDescriptions = attributeDescriptions.data()
        };
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
                .topology = vk::PrimitiveTopology::eTriangleList,
                .primitiveRestartEnable = vk::False
        };
        vk::PipelineViewportStateCreateInfo viewportState{
                .viewportCount = 1,
                .scissorCount = 1
        };
        vk::PipelineRasterizationStateCreateInfo rasterizer{
                .depthClampEnable = vk::False,
                .rasterizerDiscardEnable = vk::False,
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eBack,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .depthBiasEnable = vk::False
        };
        rasterizer.lineWidth = 1.0f;
        vk::PipelineMultisampleStateCreateInfo multisampling{
                .rasterizationSamples = vk::SampleCountFlagBits::e1,
                .sampleShadingEnable = vk::False
        };
        vk::PipelineDepthStencilStateCreateInfo depthStencil{
                .depthTestEnable = vk::True,
                .depthWriteEnable = vk::True,
                .depthCompareOp = vk::CompareOp::eLess,
                .depthBoundsTestEnable = vk::False,
                .stencilTestEnable = vk::False
        };
        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendAttachment.blendEnable = vk::False;

        vk::PipelineColorBlendStateCreateInfo colorBlending{
                .logicOpEnable = vk::False,
                .logicOp = vk::LogicOp::eCopy,
                .attachmentCount = 1,
                .pAttachments = &colorBlendAttachment
        };

        std::vector dynamicStates = {
                vk::DynamicState::eViewport,
                vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        vk::DescriptorSetLayout setLayouts[] = {*descriptorSetLayoutGlobal, *descriptorSetLayoutMaterial};

        vk::PushConstantRange pushConstantRange {
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .offset = 0,
                .size = sizeof(PushConstant)
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{  .setLayoutCount = 2, .pSetLayouts = setLayouts, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange };

        pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        vk::Format depthFormat = findDepthFormat();

        /* TASK01: Check the setup for dynamic rendering
         *
         * This new struct replaces what previously was the render pass in the pipeline creation.
         * Note how this structure is now linked in .pNext below, and .renderPass is not used.
         */
        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = depthFormat
        };

        vk::GraphicsPipelineCreateInfo pipelineInfo{
                .pNext = &pipelineRenderingCreateInfo,
                .stageCount = 2,
                .pStages = shaderStages,
                .pVertexInputState = &vertexInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &colorBlending,
                .pDynamicState = &dynamicState,
                .layout = pipelineLayout,
                .renderPass = nullptr
        };

        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{
                .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                .queueFamilyIndex = graphicsIndex
        };
        commandPool = vk::raii::CommandPool(device, poolInfo);
    }

    void createDepthResources() {
        vk::Format depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) const {
        for (const auto format : candidates) {
            vk::FormatProperties props = physicalDevice.getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    [[nodiscard]] vk::Format findDepthFormat() const {
        return findSupportedFormat(
                {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                vk::ImageTiling::eOptimal,
                vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
    }

    static bool hasStencilComponent(vk::Format format) {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    std::pair<vk::raii::Image, vk::raii::DeviceMemory> createTextureImage(const std::string& path) {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, pixels, imageSize);
        stagingBufferMemory.unmapMemory();

        stbi_image_free(pixels);

        vk::raii::Image textureImage = nullptr;
        vk::raii::DeviceMemory textureImageMemory = nullptr;

        createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        return std::make_pair(std::move(textureImage), std::move(textureImageMemory));
    }

    vk::raii::ImageView createTextureImageView(vk::raii::Image& textureImage) {
        return createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);
    }

    void createTextureSampler() {
        vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
        vk::SamplerCreateInfo samplerInfo{
                .magFilter = vk::Filter::eLinear,
                .minFilter = vk::Filter::eLinear,
                .mipmapMode = vk::SamplerMipmapMode::eLinear,
                .addressModeU = vk::SamplerAddressMode::eRepeat,
                .addressModeV = vk::SamplerAddressMode::eRepeat,
                .addressModeW = vk::SamplerAddressMode::eRepeat,
                .mipLodBias = 0.0f,
                .anisotropyEnable = vk::True,
                .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
                .compareEnable = vk::False,
                .compareOp = vk::CompareOp::eAlways
        };
        textureSampler = vk::raii::Sampler(device, samplerInfo);
    }

    vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags) {
        vk::ImageViewCreateInfo viewInfo{
                .image = image,
                .viewType = vk::ImageViewType::e2D,
                .format = format,
                .subresourceRange = { aspectFlags, 0, 1, 0, 1 }
        };
        return vk::raii::ImageView(device, viewInfo);
    }

    void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory) {
        vk::ImageCreateInfo imageInfo{
                .imageType = vk::ImageType::e2D,
                .format = format,
                .extent = {width, height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = vk::SampleCountFlagBits::e1,
                .tiling = tiling,
                .usage = usage,
                .sharingMode = vk::SharingMode::eExclusive,
                .initialLayout = vk::ImageLayout::eUndefined
        };
        image = vk::raii::Image(device, imageInfo);

        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
                .allocationSize = memRequirements.size,
                .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };
        imageMemory = vk::raii::DeviceMemory(device, allocInfo);
        image.bindMemory(imageMemory, 0);
    }

    void transitionImageLayout(const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        auto commandBuffer = beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier{
                .oldLayout = oldLayout,
                .newLayout = newLayout,
                .image = image,
                .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
        };

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask =  vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask =  vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        commandBuffer->pipelineBarrier( sourceStage, destinationStage, {}, {}, nullptr, barrier );
        endSingleTimeCommands(*commandBuffer);
    }

    void copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height) {
        std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();
        vk::BufferImageCopy region{
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
                .imageOffset = {0, 0, 0},
                .imageExtent = {width, height, 1}
        };
        commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
        endSingleTimeCommands(*commandBuffer);
    }

    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> localMaterials;
        std::string warn, err;

        if (!LoadObj(&attrib, &shapes, &localMaterials, &warn, &err, MODEL_PATH.c_str(), MODEL_PATH.substr(0, MODEL_PATH.find_last_of("/\\")).c_str())) {
            throw std::runtime_error(warn + err);
        }

        size_t materialOffset = materials.size();
        size_t oldTextureCount = textureImageViews.size();

        materials.insert(materials.end(), localMaterials.begin(), localMaterials.end());

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};
        uint32_t indexOffset = 0;

        for (const auto& shape : shapes) {
            std::cout << "Loading mesh: " << shape.name << ": " << shape.mesh.indices.size()/3 << " triangles\n";

            uint32_t startOffset = indexOffset;
            uint32_t localMaxV = 0;

            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                vertex.pos = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.texCoord = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };

                vertex.color = {1.0f, 1.0f, 1.0f};

                if (index.normal_index >= 0) {
                    vertex.normal = {
                            attrib.normals[3 * index.normal_index + 0],
                            attrib.normals[3 * index.normal_index + 1],
                            attrib.normals[3 * index.normal_index + 2]
                    };
                } else {
                    vertex.normal = {0.0f, 0.0f, 0.0f};
                }

                if (!uniqueVertices.contains(vertex)) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);

                indexOffset++;

                uint32_t vi;
                auto it = uniqueVertices.find(vertex);
                if (it != uniqueVertices.end()) {
                    vi = it->second;
                } else {
                    vi = static_cast<uint32_t>(vertices.size());
                    uniqueVertices[vertex] = vi;
                    vertices.push_back(vertex);
                }

                localMaxV = std::max(localMaxV, vi);
            }

            int localMaterialID = shape.mesh.material_ids.empty() ? -1 : shape.mesh.material_ids[0];
            int globalMaterialID = (localMaterialID < 0) ? -1 : static_cast<int>(materialOffset + localMaterialID);

            uint32_t indexCount = indexOffset - startOffset;

            // Note that this is only valid for this particular MODEL_PATH
            bool alphaCut = (shape.name.find("nettle_plant") != std::string::npos);
            bool reflective = (shape.name.find("table") != std::string::npos);

            submeshes.push_back({
                                        .indexOffset = startOffset,
                                        .indexCount = indexCount,
                                        .materialID = globalMaterialID,
                                        .firstVertex = 0u,
                                        .maxVertex = localMaxV + 1,
                                        .alphaCut = alphaCut,
                                        .reflective = reflective
                                });
        }

        for (size_t i = 0; i < localMaterials.size(); ++i) {
            const auto& material = localMaterials[i];

            if (!material.diffuse_texname.empty()) {
                std::string texturePath = MODEL_PATH.substr(0, MODEL_PATH.find_last_of("/\\")) + "/" + material.diffuse_texname;
                auto [img, mem] = createTextureImage(texturePath);
                textureImages.push_back(std::move(img));
                textureImageMemories.push_back(std::move(mem));
                textureImageViews.emplace_back(createTextureImageView(textureImages.back()));
            } else {
                std::cout << "No texture for material: " << material.name << std::endl;
            }
        }
    }

    void createVertexBuffer() {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, vertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                 vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                 vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
    }

    void createUVBuffer() {
        // Extract all texCoords into a separate vector
        std::vector<glm::vec2> uvs;
        uvs.reserve(vertices.size());
        for (auto& v: vertices) {
            uvs.push_back(v.texCoord);
        }

        vk::DeviceSize bufferSize = sizeof(uvs[0]) * uvs.size();

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, uvs.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, uvBuffer, uvBufferMemory);

        copyBuffer(stagingBuffer, uvBuffer, bufferSize);
    }

    void createInstanceLUTBuffer() {
#if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
        // TASK09: build a buffer to store the instance look-up table
        vk::DeviceSize bufferSize = sizeof(InstanceLUT) * instanceLUTs.size();

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, instanceLUTs.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal, instanceLUTBuffer, instanceLUTBufferMemory);

        copyBuffer(stagingBuffer, instanceLUTBuffer, bufferSize);
#endif // LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
    }

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer({});
            vk::raii::DeviceMemory bufferMem({});
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back( uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createAccelerationStructures() {
#if LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND
        vk::BufferDeviceAddressInfo vai{ .buffer = *vertexBuffer };
        vk::DeviceAddress vertexAddr = device.getBufferAddressKHR(vai);
        vk::BufferDeviceAddressInfo iai{ .buffer = *indexBuffer };
        vk::DeviceAddress indexAddr = device.getBufferAddressKHR(iai);

        instances.reserve(submeshes.size());
        blasBuffers.reserve(submeshes.size());
        blasMemories.reserve(submeshes.size());
        blasHandles.reserve(submeshes.size());

        vk::TransformMatrixKHR identity{};
        identity.matrix = std::array<std::array<float,4>,3>{{
            std::array<float,4>{1.f, 0.f, 0.f, 0.f},
            std::array<float,4>{0.f, 1.f, 0.f, 0.f},
            std::array<float,4>{0.f, 0.f, 1.f, 0.f}
        }};

        // TASK02: Build a bottom level acceleration structure for each submesh
        for (size_t i = 0; i < submeshes.size(); ++i) {
            const auto& submesh = submeshes[i];

            // Prepare the geometry data
            auto trianglesData = vk::AccelerationStructureGeometryTrianglesDataKHR{
                .vertexFormat = vk::Format::eR32G32B32Sfloat,
                .vertexData = vertexAddr,
                .vertexStride = sizeof(Vertex),
                .maxVertex = submesh.maxVertex,
                .indexType = vk::IndexType::eUint32,
                .indexData = indexAddr + submesh.indexOffset * sizeof(uint32_t)
            };

            vk::AccelerationStructureGeometryDataKHR geometryData(trianglesData);

            vk::AccelerationStructureGeometryKHR blasGeometry{
                .geometryType = vk::GeometryTypeKHR::eTriangles,
                .geometry = geometryData,
                .flags = vk::GeometryFlagBitsKHR::eOpaque
            };
#if LAB_TASK_LEVEL >= LAB_TASK_AS_OPAQUE_FLAG
            // TASK07
            blasGeometry.flags = (submesh.alphaCut) ? vk::GeometryFlagsKHR(0) : vk::GeometryFlagBitsKHR::eOpaque;
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_OPAQUE_FLAG

            vk::AccelerationStructureBuildGeometryInfoKHR blasBuildGeometryInfo{
                .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
                .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
                .geometryCount = 1,
                .pGeometries = &blasGeometry,
            };

            // Query the memory sizes that will be needed for this BLAS
            auto primitiveCount = static_cast<uint32_t>(submesh.indexCount / 3);

            vk::AccelerationStructureBuildSizesInfoKHR blasBuildSizes =
                device.getAccelerationStructureBuildSizesKHR(
                    vk::AccelerationStructureBuildTypeKHR::eDevice,
                    blasBuildGeometryInfo,
                    { primitiveCount }
            );

            // Create a scratch buffer for the BLAS, this will hold temporary data
            // during the build process
            vk::raii::Buffer scratchBuffer = nullptr;
            vk::raii::DeviceMemory scratchMemory = nullptr;
            createBuffer(blasBuildSizes.buildScratchSize,
                         vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress,
                         vk::MemoryPropertyFlagBits::eDeviceLocal,
                         scratchBuffer, scratchMemory);

            // Save the scratch buffer address in the build info structure
            vk::BufferDeviceAddressInfo scratchAddressInfo{ .buffer = *scratchBuffer };
            vk::DeviceAddress scratchAddr = device.getBufferAddressKHR(scratchAddressInfo);
            blasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;

            // Create a buffer for the BLAS itself now that we now the required size
            vk::raii::Buffer blasBuffer = nullptr;
            vk::raii::DeviceMemory blasMemory = nullptr;
            blasBuffers.emplace_back(std::move(blasBuffer));
            blasMemories.emplace_back(std::move(blasMemory));
            createBuffer(blasBuildSizes.accelerationStructureSize,
                         vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress |
                         vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                         vk::MemoryPropertyFlagBits::eDeviceLocal,
                         blasBuffers[i], blasMemories[i]);

            // Create and store the BLAS handle
            vk::AccelerationStructureCreateInfoKHR blasCreateInfo{
                .buffer = blasBuffers[i],
                .offset = 0,
                .size = blasBuildSizes.accelerationStructureSize,
                .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
            };

            blasHandles.emplace_back(device.createAccelerationStructureKHR(blasCreateInfo));

            // Save the BLAS handle in the build info structure
            blasBuildGeometryInfo.dstAccelerationStructure = blasHandles[i];

            // Prepare the build range for the BLAS
            vk::AccelerationStructureBuildRangeInfoKHR blasRangeInfo{
                .primitiveCount = primitiveCount,
                .primitiveOffset = 0,
                .firstVertex = submesh.firstVertex,
                .transformOffset = 0
            };

            // Build the BLAS
            auto cmd = beginSingleTimeCommands();
            cmd->buildAccelerationStructuresKHR({ blasBuildGeometryInfo }, { &blasRangeInfo });
            endSingleTimeCommands(*cmd);

            // TASK03: Create a BLAS instance for the TLAS
            vk::AccelerationStructureDeviceAddressInfoKHR addrInfo{
                .accelerationStructure = *blasHandles[i]
            };
            vk::DeviceAddress blasDeviceAddr = device.getAccelerationStructureAddressKHR(addrInfo);

            vk::AccelerationStructureInstanceKHR instance{
                .transform = identity,
                .mask = 0xFF,
                .accelerationStructureReference = blasDeviceAddr
            };

            instances.push_back(instance);

#if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
            // TASK09: store the instance look-up table entry
            instances[i].instanceCustomIndex = static_cast<uint32_t>(i);

            instanceLUTs.push_back({ static_cast<uint32_t>(submesh.materialID), submesh.indexOffset });
#endif // LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
        }

        // TASK03: Prepare the instance data buffer
        vk::DeviceSize instBufferSize = sizeof(instances[0]) * instances.size();
        createBuffer(instBufferSize,
                     vk::BufferUsageFlagBits::eShaderDeviceAddress |
                     vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                     instanceBuffer, instanceMemory);

        void *ptr = instanceMemory.mapMemory(0, instBufferSize);
        memcpy(ptr, instances.data(), instBufferSize);
        instanceMemory.unmapMemory();

        vk::BufferDeviceAddressInfo instanceAddrInfo{ .buffer = instanceBuffer };
        vk::DeviceAddress instanceAddr = device.getBufferAddressKHR(instanceAddrInfo);

        // Prepare the geometry (instance) data
        auto instancesData = vk::AccelerationStructureGeometryInstancesDataKHR{
            .arrayOfPointers = vk::False,
            .data = instanceAddr
        };

        vk::AccelerationStructureGeometryDataKHR geometryData(instancesData);

        vk::AccelerationStructureGeometryKHR tlasGeometry{
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .geometry = geometryData
        };

        vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildGeometryInfo{
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &tlasGeometry
        };

#if LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION
        tlasBuildGeometryInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION

        // Query the memory sizes that will be needed for this TLAS
        auto primitiveCount = static_cast<uint32_t>(instances.size());

        vk::AccelerationStructureBuildSizesInfoKHR tlasBuildSizes =
            device.getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice,
                tlasBuildGeometryInfo,
                { primitiveCount }
        );

        // Create a scratch buffer for the TLAS, this will hold temporary data
        // during the build process
        createBuffer(
            tlasBuildSizes.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            tlasScratchBuffer, tlasScratchMemory
        );

        // Save the scratch buffer address in the build info structure
        vk::BufferDeviceAddressInfo scratchAddressInfo{ .buffer = *tlasScratchBuffer };
        vk::DeviceAddress scratchAddr = device.getBufferAddressKHR(scratchAddressInfo);
        tlasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;

        // Create a buffer for the TLAS itself now that we now the required size
        createBuffer(
            tlasBuildSizes.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            tlasBuffer, tlasMemory
        );

        // Create and store the TLAS handle
        vk::AccelerationStructureCreateInfoKHR tlasCreateInfo{
            .buffer = tlasBuffer,
            .offset = 0,
            .size = tlasBuildSizes.accelerationStructureSize,
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        };

        tlas = device.createAccelerationStructureKHR(tlasCreateInfo);

        // Save the TLAS handle in the build info structure
        tlasBuildGeometryInfo.dstAccelerationStructure = tlas;

         // Prepare the build range for the TLAS
         vk::AccelerationStructureBuildRangeInfoKHR tlasRangeInfo{
             .primitiveCount = primitiveCount,
             .primitiveOffset = 0,
             .firstVertex = 0,
             .transformOffset = 0
         };

        // Build the TLAS
        auto cmd = beginSingleTimeCommands();

        cmd->buildAccelerationStructuresKHR({ tlasBuildGeometryInfo }, { &tlasRangeInfo });

        endSingleTimeCommands(*cmd);
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND
    }

    void createDescriptorPool() {
        std::array poolSize {
                vk::DescriptorPoolSize( vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
                vk::DescriptorPoolSize( vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT),
                vk::DescriptorPoolSize( vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * 3), // indices, UVs, instance LUT
                vk::DescriptorPoolSize( vk::DescriptorType::eSampler, MAX_FRAMES_IN_FLIGHT),
                vk::DescriptorPoolSize( vk::DescriptorType::eSampledImage, (uint32_t)materials.size())
        };
        vk::DescriptorPoolCreateInfo poolInfo{
                .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
                         vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
                .maxSets = MAX_FRAMES_IN_FLIGHT + 1, // + 1 for bindless materials
                .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
                .pPoolSizes = poolSize.data()
        };
        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createDescriptorSets() {
        // Global descriptor sets (per frame)
        std::vector<vk::DescriptorSetLayout> globalLayouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayoutGlobal);

        vk::DescriptorSetAllocateInfo allocInfoGlobal{
                .descriptorPool = descriptorPool,
                .descriptorSetCount = static_cast<uint32_t>(globalLayouts.size()),
                .pSetLayouts = globalLayouts.data()
        };

        globalDescriptorSets.clear();
        globalDescriptorSets = device.allocateDescriptorSets(allocInfoGlobal);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            // Uniform buffer
            vk::DescriptorBufferInfo bufferInfo{
                    .buffer = uniformBuffers[i],
                    .offset = 0,
                    .range = sizeof(UniformBufferObject)
            };

            vk::WriteDescriptorSet bufferWrite{
                    .dstSet = globalDescriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &bufferInfo
            };

#if LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND
            // TASK04: define the acceleration structure descriptor.
            vk::WriteDescriptorSetAccelerationStructureKHR asInfo{
                .accelerationStructureCount = 1,
                .pAccelerationStructures = {&*tlas}
            };

            vk::WriteDescriptorSet asWrite{
                .pNext = &asInfo,
                .dstSet = globalDescriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eAccelerationStructureKHR
            };
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND

            // Indices SSBO
            vk::DescriptorBufferInfo indexBufferInfo{
                    .buffer = indexBuffer,
                    .offset = 0,
                    .range = sizeof(uint32_t) * indices.size()
            };

            vk::WriteDescriptorSet indexBufferWrite{
                    .dstSet = globalDescriptorSets[i],
                    .dstBinding = 2,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &indexBufferInfo
            };

            // UVs SSBO
            vk::DescriptorBufferInfo uvBufferInfo{
                    .buffer = uvBuffer,
                    .offset = 0,
                    .range = sizeof(glm::vec2) * vertices.size()
            };

            vk::WriteDescriptorSet uvBufferWrite{
                    .dstSet = globalDescriptorSets[i],
                    .dstBinding = 3,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &uvBufferInfo
            };

#if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
            // TASK09: Instance LUT SSBO
            vk::DescriptorBufferInfo instanceLUTBufferInfo{
                .buffer = instanceLUTBuffer,
                .offset = 0,
                .range = sizeof(InstanceLUT) * instanceLUTs.size()
            };

            vk::WriteDescriptorSet instanceLUTBufferWrite{
                .dstSet = globalDescriptorSets[i],
                .dstBinding = 4,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &instanceLUTBufferInfo
            };
#endif // LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT

#if LAB_TASK_LEVEL >= LAB_TASK_INSTANCE_LUT
            // TASK09: Include the instance look-up table descriptor
            std::array<vk::WriteDescriptorSet, 5> descriptorWrites{bufferWrite, asWrite, indexBufferWrite, uvBufferWrite, instanceLUTBufferWrite};
#elif LAB_TASK_LEVEL >= LAB_TASK_AS_BUILD_AND_BIND
            // TASK04: Include the acceleration structure descriptor
            std::array<vk::WriteDescriptorSet, 4> descriptorWrites{bufferWrite, asWrite, indexBufferWrite, uvBufferWrite};
#else
            std::array<vk::WriteDescriptorSet, 3> descriptorWrites{bufferWrite, indexBufferWrite, uvBufferWrite};
#endif

            device.updateDescriptorSets(descriptorWrites, {});
        }

        // Material descriptor sets (per material)
        std::vector<uint32_t> variableCounts = { static_cast<uint32_t>(textureImageViews.size()) };
        vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo{
                .descriptorSetCount = 1,
                .pDescriptorCounts = variableCounts.data()
        };

        std::vector<vk::DescriptorSetLayout> layouts{ *descriptorSetLayoutMaterial };

        vk::DescriptorSetAllocateInfo allocInfo {
                .pNext = &variableCountInfo,
                .descriptorPool = descriptorPool,
                .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
                .pSetLayouts = layouts.data()
        };

        materialDescriptorSets = device.allocateDescriptorSets(allocInfo);

        // Sampler
        vk::DescriptorImageInfo samplerInfo{
                .sampler = textureSampler
        };

        vk::WriteDescriptorSet samplerWrite{
                .dstSet = materialDescriptorSets[0],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eSampler,
                .pImageInfo = &samplerInfo
        };

        device.updateDescriptorSets({samplerWrite}, {});

        // Textures
        std::vector<vk::DescriptorImageInfo> imageInfos;
        imageInfos.reserve(textureImageViews.size());
        for (auto& iv : textureImageViews) {
            vk::DescriptorImageInfo imageInfo{
                    .imageView = iv,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };
            imageInfos.push_back(imageInfo);
        }

        vk::WriteDescriptorSet materialWrite{
                .dstSet = materialDescriptorSets[0],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = static_cast<uint32_t>(imageInfos.size()),
                .descriptorType = vk::DescriptorType::eSampledImage,
                .pImageInfo = imageInfos.data()
        };

        device.updateDescriptorSets({materialWrite}, {});
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {
        vk::BufferCreateInfo bufferInfo{
                .size = size,
                .usage = usage,
                .sharingMode = vk::SharingMode::eExclusive
        };
        buffer = vk::raii::Buffer(device, bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
                .allocationSize = memRequirements.size,
                .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };
        vk::MemoryAllocateFlagsInfo allocFlagsInfo{};
        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            allocFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
            allocInfo.pNext = &allocFlagsInfo;
        }
        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(bufferMemory, 0);
    }

    std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo{
                .commandPool = commandPool,
                .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = 1
        };
        std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

        vk::CommandBufferBeginInfo beginInfo{
                .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };
        commandBuffer->begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const {
        commandBuffer.end();

        vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer };
        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    void copyBuffer(vk::raii::Buffer & srcBuffer, vk::raii::Buffer & dstBuffer, vk::DeviceSize size) {
        vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
        vk::raii::CommandBuffer commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
        commandCopyBuffer.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{ .size = size });
        commandCopyBuffer.end();
        graphicsQueue.submit(vk::SubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer }, nullptr);
        graphicsQueue.waitIdle();
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = MAX_FRAMES_IN_FLIGHT };
        commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void recordCommandBuffer(uint32_t imageIndex) {
        commandBuffers[currentFrame].begin({});
        // Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
        transition_image_layout(
                imageIndex,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {},                                                     // srcAccessMask (no need to wait for previous operations)
                vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
                vk::PipelineStageFlagBits2::eTopOfPipe,                   // srcStage
                vk::PipelineStageFlagBits2::eColorAttachmentOutput        // dstStage
        );
        // Transition depth image to depth attachment optimal layout
        vk::ImageMemoryBarrier2 depthBarrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
                .srcAccessMask = {},
                .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = depthImage,
                .subresourceRange = {
                        .aspectMask = vk::ImageAspectFlagBits::eDepth,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };
        vk::DependencyInfo depthDependencyInfo = {
                .dependencyFlags = {},
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &depthBarrier
        };
        commandBuffers[currentFrame].pipelineBarrier2(depthDependencyInfo);

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
        vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

        /* TASK01: Check the setup for dynamic rendering
         *
         * With dynamic rendering, we specify the image view and load/store operations directly
         * in the vk::RenderingAttachmentInfo structure.
         * This approach eliminates the need for explicit render pass and framebuffer objects,
         * simplifying the code and providing flexibility to change attachments at runtime.
         */

        vk::RenderingAttachmentInfo colorAttachmentInfo = {
                .imageView = swapChainImageViews[imageIndex],
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = clearColor
        };

        vk::RenderingAttachmentInfo depthAttachmentInfo = {
                .imageView = depthImageView,
                .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eDontCare,
                .clearValue = clearDepth
        };

        // The vk::RenderingInfo structure combines these attachments with other rendering parameters.
        vk::RenderingInfo renderingInfo = {
                .renderArea = { .offset = { 0, 0 }, .extent = swapChainExtent },
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &colorAttachmentInfo,
                .pDepthAttachment = &depthAttachmentInfo
        };

        // Note: .beginRendering replaces the previous .beginRenderPass call.
        commandBuffers[currentFrame].beginRendering(renderingInfo);

        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
        commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
        commandBuffers[currentFrame].bindVertexBuffers(0, *vertexBuffer, {0});
        commandBuffers[currentFrame].bindIndexBuffer( *indexBuffer, 0, vk::IndexType::eUint32 );
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *globalDescriptorSets[currentFrame], nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, *materialDescriptorSets[0], nullptr);

        for (auto& sub : submeshes) {
            // TASK09: Bindless resources
            PushConstant pushConstant = {
                    .materialIndex = sub.materialID < 0 ? 0u : static_cast<uint32_t>(sub.materialID),
#if LAB_TASK_LEVEL >= LAB_TASK_REFLECTIONS
                    .reflective = sub.reflective
#endif // LAB_TASK_LEVEL >= LAB_TASK_REFLECTIONS
            };
            commandBuffers[currentFrame].pushConstants<PushConstant>(pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, pushConstant);

            commandBuffers[currentFrame].drawIndexed(sub.indexCount, 1, sub.indexOffset, 0, 0);
        }

        commandBuffers[currentFrame].endRendering();

        // After rendering, transition the swapchain image to PRESENT_SRC
        transition_image_layout(
                imageIndex,
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageLayout::ePresentSrcKHR,
                vk::AccessFlagBits2::eColorAttachmentWrite,            // srcAccessMask
                {},                                                    // dstAccessMask
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,    // srcStage
                vk::PipelineStageFlagBits2::eBottomOfPipe              // dstStage
        );

        commandBuffers[currentFrame].end();
    }

    void transition_image_layout(
            uint32_t imageIndex,
            vk::ImageLayout old_layout,
            vk::ImageLayout new_layout,
            vk::AccessFlags2 src_access_mask,
            vk::AccessFlags2 dst_access_mask,
            vk::PipelineStageFlags2 src_stage_mask,
            vk::PipelineStageFlags2 dst_stage_mask
    ) {
        vk::ImageMemoryBarrier2 barrier = {
                .srcStageMask = src_stage_mask,
                .srcAccessMask = src_access_mask,
                .dstStageMask = dst_stage_mask,
                .dstAccessMask = dst_access_mask,
                .oldLayout = old_layout,
                .newLayout = new_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapChainImages[imageIndex],
                .subresourceRange = {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };
        vk::DependencyInfo dependency_info = {
                .dependencyFlags = {},
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrier
        };
        commandBuffers[currentFrame].pipelineBarrier2(dependency_info);
    }

    void createSyncObjects() {
        presentCompleteSemaphore.clear();
        renderFinishedSemaphore.clear();
        inFlightFences.clear();

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            presentCompleteSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
            renderFinishedSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
        }


        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            inFlightFences.emplace_back(device, vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(currentTime - startTime).count();

        auto eye = glm::vec3(2.0f, 2.0f, 2.0f);

        ubo.model = rotate(glm::mat4(1.0f), time * 0.1f * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = lookAt(eye, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        ubo.cameraPos = eye;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

#if LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION
    void updateTopLevelAS(const glm::mat4 & model) {
        vk::TransformMatrixKHR tm{};
        auto &M = model;
        tm.matrix = std::array<std::array<float,4>,3>{{
            std::array<float,4>{M[0][0], M[1][0], M[2][0], M[3][0]},
            std::array<float,4>{M[0][1], M[1][1], M[2][1], M[3][1]},
            std::array<float,4>{M[0][2], M[1][2], M[2][2], M[3][2]}
        }};

        // TASK06: update the instances to use the new transform matrix.
        for (auto & instance : instances) {
            instance.setTransform(tm);
        }

        auto primitiveCount = static_cast<uint32_t>(instances.size());
        vk::DeviceSize instBufferSize = sizeof(instances[0]) * primitiveCount;

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(instBufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, instBufferSize);
        memcpy(dataStaging, instances.data(), instBufferSize);
        stagingBufferMemory.unmapMemory();

        copyBuffer(stagingBuffer, instanceBuffer, instBufferSize);

        vk::BufferDeviceAddressInfo instanceAddrInfo{ .buffer = instanceBuffer };
        vk::DeviceAddress instanceAddr = device.getBufferAddressKHR(instanceAddrInfo);

        // Prepare the geometry (instance) data
        auto instancesData = vk::AccelerationStructureGeometryInstancesDataKHR{
            .arrayOfPointers = vk::False,
            .data = instanceAddr
        };

        vk::AccelerationStructureGeometryDataKHR geometryData(instancesData);

        vk::AccelerationStructureGeometryKHR tlasGeometry{
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .geometry = geometryData
        };

        // TASK06: Note the new parameters to re-build the TLAS in-place
        vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildGeometryInfo{
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
            .mode = vk::BuildAccelerationStructureModeKHR::eUpdate,
            .srcAccelerationStructure = tlas,
            .dstAccelerationStructure = tlas,
            .geometryCount = 1,
            .pGeometries = &tlasGeometry
        };

        vk::BufferDeviceAddressInfo scratchAddressInfo{ .buffer = *tlasScratchBuffer };
        vk::DeviceAddress scratchAddr = device.getBufferAddressKHR(scratchAddressInfo);
        tlasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;

        // Prepare the build range for the TLAS
        vk::AccelerationStructureBuildRangeInfoKHR tlasRangeInfo{
            .primitiveCount = primitiveCount,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0
        };

        // Re-build the TLAS
        auto cmd = beginSingleTimeCommands();

        // Pre-build barrier
        vk::MemoryBarrier preBarrier {
            .srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR | vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderRead,
            .dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR | vk::AccessFlagBits::eAccelerationStructureWriteKHR
        };

        cmd->pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eFragmentShader, // srcStageMask
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, // dstStageMask
            {}, // dependencyFlags
            preBarrier, // memoryBarriers
            {}, // bufferMemoryBarriers
            {} // imageMemoryBarriers
        );

        cmd->buildAccelerationStructuresKHR({ tlasBuildGeometryInfo }, { &tlasRangeInfo });

        // Post-build barrier
        vk::MemoryBarrier postBarrier {
            .srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR,
            .dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR | vk::AccessFlagBits::eShaderRead
        };

        cmd->pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, // srcStageMask
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits::eFragmentShader, // dstStageMask
            {}, // dependencyFlags
            postBarrier, // memoryBarriers
            {}, // bufferMemoryBarriers
            {} // imageMemoryBarriers
        );

        endSingleTimeCommands(*cmd);
    }
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION

    void drawFrame() {
        while ( vk::Result::eTimeout == device.waitForFences( *inFlightFences[currentFrame], vk::True, UINT64_MAX ) )
            ;
        auto [result, imageIndex] = swapChain.acquireNextImage( UINT64_MAX, *presentCompleteSemaphore[semaphoreIndex], nullptr );

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        updateUniformBuffer(currentFrame);
#if LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION
        // TASK06: Update the TLAS with the current model matrix
        updateTopLevelAS(ubo.model);
#endif // LAB_TASK_LEVEL >= LAB_TASK_AS_ANIMATION

        device.resetFences(  *inFlightFences[currentFrame] );
        commandBuffers[currentFrame].reset();
        recordCommandBuffer(imageIndex);

        vk::PipelineStageFlags waitDestinationStageMask( vk::PipelineStageFlagBits::eColorAttachmentOutput );
        const vk::SubmitInfo submitInfo{ .waitSemaphoreCount = 1, .pWaitSemaphores = &*presentCompleteSemaphore[semaphoreIndex],
                .pWaitDstStageMask = &waitDestinationStageMask, .commandBufferCount = 1, .pCommandBuffers = &*commandBuffers[currentFrame],
                .signalSemaphoreCount = 1, .pSignalSemaphores = &*renderFinishedSemaphore[imageIndex] };
        graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);


        const vk::PresentInfoKHR presentInfoKHR{ .waitSemaphoreCount = 1, .pWaitSemaphores = &*renderFinishedSemaphore[imageIndex],
                .swapchainCount = 1, .pSwapchains = &*swapChain, .pImageIndices = &imageIndex };
        result = presentQueue.presentKHR(presentInfoKHR);
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }
        semaphoreIndex = (semaphoreIndex + 1) % presentCompleteSemaphore.size();
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{ .codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data()) };
        vk::raii::ShaderModule shaderModule{ device, createInfo };

        return shaderModule;
    }

    static vk::Format chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        const auto formatIt = std::ranges::find_if(availableFormats,
                                                   [](const auto& format) {
                                                       return format.format == vk::Format::eB8G8R8A8Srgb &&
                                                              format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
                                                   });
        return formatIt != availableFormats.end() ? formatIt->format : availableFormats[0].format;
    }

    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        return std::ranges::any_of(availablePresentModes,
                                   [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; } ) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        return {
                std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    [[nodiscard]] std::vector<const char*> getRequiredExtensions() const {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers) {
            extensions.push_back(vk::EXTDebugUtilsExtensionName );
        }

        return extensions;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
        if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
            std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
        }

        return vk::False;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        std::vector<char> buffer(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();
        return buffer;
    }
};

int main() {
    try {
        HelloTriangleApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

#endif
