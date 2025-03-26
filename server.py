import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from concurrent import futures

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math
import grpc

from concurrent import futures
import time
import generated.sph_optimized_pb2 as sph_optimized_pb2
import generated.sph_optimized_pb2_grpc as sph_optimized_pb2_grpc
import threading 

from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime


cuda_kernel_code = """
#include <math.h>

// Константы для симуляции
#define PI 3.141592653589f
#define REST_DENSITY 1000.0f
#define GAS_CONSTANT 2000.0f 
#define VISCOSITY 10.0f  

#define SMOOTHING_LENGTH .5f
#define GRAVITY_X 0.0f
#define GRAVITY_Y 0.0f
#define GRAVITY_Z -9.8f
#define EPSILON 1e-6f

#define BOUNDARY 10.0f 

typedef struct {
    float3 pos;
    float3 vel;
    float mass;
    float density;
    float pressure;
} Particle;

/*
 * Вычисление значения ядра сглаживания (кубический сплайн)
 * r: расстояние между частицами
 * h: радиус сглаживания
 */
__device__ float cubic_spline_kernel(float r, float h) {
    float q = r / h;
    float result = 0.0f;
    
    if (q < 1.0f) {
        result = 1.0f - 1.5f * q * q + 0.75f * q * q * q;
    } else if (q < 2.0f) {
        float tmp = 2.0f - q;
        result = 0.25f * tmp * tmp * tmp;
    }
    
    float factor = 8.0f / (PI * h * h * h);
    return factor * result;
}

/*
 * Вычисление градиента ядра сглаживания
 * r: вектор от j к i (r_i - r_j)
 * r_len: длина вектора r
 * h: радиус сглаживания
 */
__device__ float3 cubic_spline_kernel_gradient(float3 r, float r_len, float h) {
    if (r_len < EPSILON) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    float q = r_len / h;
    float derivative = 0.0f;
    
    if (q < 1.0f) {
        derivative = q * (2.25f * q - 3.0f);
    } else if (q < 2.0f) {
        float tmp = 2.0f - q;
        derivative = -0.75f * tmp * tmp;
    }
    
    float factor = 8.0f / (PI * h * h * h * h);
    
    float3 result;
    float scale = factor * derivative / (r_len + EPSILON);
    result.x = r.x * scale;
    result.y = r.y * scale;
    result.z = r.z * scale;
    
    return result;
}

/*
 * Построение сетки для поиска соседей
 * Каждая частица помещается в ячейку сетки
 */
__global__ void build_hash_grid(
    int *cell_start,
    int *cell_end,
    int *particle_indices,
    float *positions_x,
    float *positions_y,
    float *positions_z,
    int num_particles,
    float cell_size,
    int grid_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_particles) return;
    
    // Позиция частицы
    float px = positions_x[tid];
    float py = positions_y[tid];
    float pz = positions_z[tid];
    
    // Вычисляем индекс ячейки
    int cx = (int)floorf((px + BOUNDARY) / cell_size);
    int cy = (int)floorf((py + BOUNDARY) / cell_size);
    int cz = (int)floorf((pz + BOUNDARY) / cell_size);
    
    // Ограничиваем индексы сетки
    cx = max(0, min(cx, grid_size - 1));
    cy = max(0, min(cy, grid_size - 1));
    cz = max(0, min(cz, grid_size - 1));
    
    // Линейный индекс ячейки
    int cell_idx = cx + cy * grid_size + cz * grid_size * grid_size;

    // Получаем «offset» внутри ячейки при помощи атомарного прибавления
    // (cell_end хранит, сколько частиц уже занесено в эту ячейку)
    int offset = atomicAdd(&cell_end[cell_idx], 1);

    // Сохраняем реальный индекс tid частицы в particle_indices[offset]
    // (частицы фактически сортируются по ячейкам)
    particle_indices[offset] = tid;

    // Если ячейка раньше была пустая (offset == 0), запоминаем,
    // где начинаются индексы в particle_indices для этой ячейки
    if (offset == 0) {
        cell_start[cell_idx] = offset;
    }
}

/*
 * Вычисление плотности с использованием сетки для поиска соседей
 * Использует разделяемую память для кэширования данных блока
 */
__global__ void compute_density(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    float *masses,
    float *densities,
    int *cell_start, // исправленная логика
    int *cell_end, // здесь хранится count
    int *particle_indices, // теперь нужен для извлечения neighbor_tid
    int num_particles,
    float cell_size,
    int grid_size
    ) {
    // Используем разделяемую память для кэширования позиций и масс
    extern __shared__ float shared_data[];
    float *shared_pos_x = shared_data;
    float *shared_pos_y = &shared_data[blockDim.x];
    float *shared_pos_z = &shared_data[2 * blockDim.x];
    float *shared_mass = &shared_data[3 * blockDim.x];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    if (tid < num_particles) {
        // Загружаем данные в разделяемую память
        shared_pos_x[local_tid] = positions_x[tid];
        shared_pos_y[local_tid] = positions_y[tid];
        shared_pos_z[local_tid] = positions_z[tid];
        shared_mass[local_tid] = masses[tid];
    }
    __syncthreads();

    if (tid >= num_particles) return;

    float px = shared_pos_x[local_tid];
    float py = shared_pos_y[local_tid];
    float pz = shared_pos_z[local_tid];
    float mass_i = shared_mass[local_tid];
    float density = 0.0f;

    // Индекс ячейки для текущей частицы
    int cx = (int)floorf((px + BOUNDARY) / cell_size);
    int cy = (int)floorf((py + BOUNDARY) / cell_size);
    int cz = (int)floorf((pz + BOUNDARY) / cell_size);

    cx = max(0, min(cx, grid_size - 1));
    cy = max(0, min(cy, grid_size - 1));
    cz = max(0, min(cz, grid_size - 1));

    for (int oz = -1; oz <= 1; oz++) {
        int cz_o = cz + oz;
        if (cz_o < 0 || cz_o >= grid_size) continue;
        
        for (int oy = -1; oy <= 1; oy++) {
            int cy_o = cy + oy;
            if (cy_o < 0 || cy_o >= grid_size) continue;
            
            for (int ox = -1; ox <= 1; ox++) {
                int cx_o = cx + ox;
                if (cx_o < 0 || cx_o >= grid_size) continue;
                
                int cell_idx = cx_o + cy_o * grid_size + cz_o * grid_size * grid_size;
                
                int start_idx = cell_start[cell_idx];
                if (start_idx == -1) {
                    // Ячейка пуста
                    continue;
                }
                int count = cell_end[cell_idx];  // сколько частиц в этой ячейке
                
                // Перебираем частицы в ячейке
                for (int k = start_idx; k < start_idx + count; k++) {
                    // Извлекаем реальный tid соседа
                    int neighbor_tid = particle_indices[k];
                    
                    // Если сосед == мы сами, можем пропустить (необязательно)
                    if (neighbor_tid == tid) continue;
                    
                    // Если сосед в этом же блоке, используем shared
                    float qx, qy, qz, qm;
                    int block_start_tid = blockIdx.x * blockDim.x;
                    int block_end_tid = block_start_tid + blockDim.x;
                    
                    if (neighbor_tid >= block_start_tid && neighbor_tid < block_end_tid) {
                        int sj = neighbor_tid - block_start_tid;
                        qx = shared_pos_x[sj];
                        qy = shared_pos_y[sj];
                        qz = shared_pos_z[sj];
                        qm = shared_mass[sj];
                    } else {
                        qx = positions_x[neighbor_tid];
                        qy = positions_y[neighbor_tid];
                        qz = positions_z[neighbor_tid];
                        qm = masses[neighbor_tid];
                    }
                    
                    // Считаем расстояние
                    float rx = px - qx;
                    float ry = py - qy;
                    float rz = pz - qz;
                    float r = sqrtf(rx*rx + ry*ry + rz*rz);
                    
                    if (r < 2.0f * SMOOTHING_LENGTH) {
                        float kernel_value = cubic_spline_kernel(r, SMOOTHING_LENGTH);
                        density += qm * kernel_value;
                    }
                }
            }
        }
    }

    // Учёт «self-contribution»
    density += mass_i * cubic_spline_kernel(0.0f, SMOOTHING_LENGTH);

    densities[tid] = density;
}

/*
 * Вычисление давления из плотности
 */
__global__ void compute_pressure(
    float *densities,
    float *pressures,
    int num_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;
    
    // Параметры с ограничениями
    const float B = GAS_CONSTANT;  // Уменьшенная газовая постоянная
    const float rho_0 = REST_DENSITY;
 
    // Ограничение плотности и плавная коррекция
    float rho = densities[tid];
    float density_ratio = rho / rho_0;
    
    // Сглаженное уравнение состояния
    pressures[tid] = B * (density_ratio - 1.0f);
}

/*
 * Вычисление сил и ускорений
 * Использует сетку для эффективного поиска соседей
 */
__global__ void compute_acceleration(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    float *velocities_x,
    float *velocities_y,
    float *velocities_z,
    float *masses,
    float *densities,
    float *pressures,
    float *accelerations_x,
    float *accelerations_y,
    float *accelerations_z,
    int *cell_start,
    int *cell_end,
    int *particle_indices,
    int num_particles,
    float cell_size,
    int grid_size
) {
    // Используем разделяемую память для кэширования данных блока
    extern __shared__ float shared_acc_data[];
    float *shared_pos_x = shared_acc_data;
    float *shared_pos_y = &shared_acc_data[blockDim.x];
    float *shared_pos_z = &shared_acc_data[2 * blockDim.x];
    float *shared_vel_x = &shared_acc_data[3 * blockDim.x];
    float *shared_vel_y = &shared_acc_data[4 * blockDim.x];
    float *shared_vel_z = &shared_acc_data[5 * blockDim.x];
    float *shared_mass = &shared_acc_data[6 * blockDim.x];
    float *shared_dens = &shared_acc_data[7 * blockDim.x];
    float *shared_pres = &shared_acc_data[8 * blockDim.x];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    if (tid < num_particles) {
        shared_pos_x[local_tid] = positions_x[tid];
        shared_pos_y[local_tid] = positions_y[tid];
        shared_pos_z[local_tid] = positions_z[tid];
        shared_vel_x[local_tid] = velocities_x[tid];
        shared_vel_y[local_tid] = velocities_y[tid];
        shared_vel_z[local_tid] = velocities_z[tid];
        shared_mass[local_tid]  = masses[tid];
        shared_dens[local_tid]  = densities[tid];
        shared_pres[local_tid]  = pressures[tid];
    }
    __syncthreads();

    if (tid >= num_particles) return;

    // Собственные данные частицы i
    float px = shared_pos_x[local_tid];
    float py = shared_pos_y[local_tid];
    float pz = shared_pos_z[local_tid];
    float vx = shared_vel_x[local_tid];
    float vy = shared_vel_y[local_tid];
    float vz = shared_vel_z[local_tid];
    float density_i = shared_dens[local_tid];
    float pressure_i = shared_pres[local_tid];

    // Начальное ускорение (например, можно сразу ax = GRAVITY_X и т. д.)
    float ax = GRAVITY_X;
    float ay = GRAVITY_Y;
    float az = GRAVITY_Z;

    // Находим, в какой ячейке сидит наша частица
    int cx = (int)floorf((px + BOUNDARY) / cell_size);
    int cy = (int)floorf((py + BOUNDARY) / cell_size);
    int cz = (int)floorf((pz + BOUNDARY) / cell_size);

    cx = max(0, min(cx, grid_size - 1));
    cy = max(0, min(cy, grid_size - 1));
    cz = max(0, min(cz, grid_size - 1));

    float3 vel_avg = {0.0f, 0.0f, 0.0f};

    for (int oz = -1; oz <= 1; oz++) {
        int cz_o = cz + oz;
        if (cz_o < 0 || cz_o >= grid_size) continue;
        
        for (int oy = -1; oy <= 1; oy++) {
            int cy_o = cy + oy;
            if (cy_o < 0 || cy_o >= grid_size) continue;
            
            for (int ox = -1; ox <= 1; ox++) {
                int cx_o = cx + ox;
                if (cx_o < 0 || cx_o >= grid_size) continue;
                
                int cell_idx = cx_o + cy_o * grid_size + cz_o * grid_size * grid_size;
                
                int start_idx = cell_start[cell_idx];
                if (start_idx == -1) {
                    // Пустая ячейка
                    continue;
                }
                int count = cell_end[cell_idx];
                
                for (int k = start_idx; k < start_idx + count; k++) {
                    int neighbor_tid = particle_indices[k];
                    if (neighbor_tid == tid) continue; // пропускаем себя
                    
                    // Извлекаем из памяти данные соседа
                    float qx, qy, qz, qvx, qvy, qvz, qm, qd, qp;
                    
                    int block_start = blockIdx.x * blockDim.x;
                    int block_end   = block_start + blockDim.x;
                    
                    if (neighbor_tid >= block_start && neighbor_tid < block_end) {
                        int sj = neighbor_tid - block_start;
                        qx = shared_pos_x[sj];
                        qy = shared_pos_y[sj];
                        qz = shared_pos_z[sj];
                        qvx = shared_vel_x[sj];
                        qvy = shared_vel_y[sj];
                        qvz = shared_vel_z[sj];
                        qm  = shared_mass[sj];
                        qd  = shared_dens[sj];
                        qp  = shared_pres[sj];
                    } else {
                        qx = positions_x[neighbor_tid];
                        qy = positions_y[neighbor_tid];
                        qz = positions_z[neighbor_tid];
                        qvx = velocities_x[neighbor_tid];
                        qvy = velocities_y[neighbor_tid];
                        qvz = velocities_z[neighbor_tid];
                        qm  = masses[neighbor_tid];
                        qd  = densities[neighbor_tid];
                        qp  = pressures[neighbor_tid];
                    }
                    
                    // Направление i <- j
                    float rx = px - qx;
                    float ry = py - qy;
                    float rz = pz - qz;
                    float r  = sqrtf(rx*rx + ry*ry + rz*rz);
                    
                    if (r < 2.f * SMOOTHING_LENGTH && qd > EPSILON) {
                        
                        float3 r_vec = make_float3(rx, ry, rz);
                        
                        // Градиент ядра
                        float3 gradW = cubic_spline_kernel_gradient(r_vec, r, SMOOTHING_LENGTH);
                        
                        // Давление (симметричная формула)
                        float pressure_term = -qm * (pressure_i/(density_i*density_i) + qp/(qd*qd));
                        ax += pressure_term * gradW.x;
                        ay += pressure_term * gradW.y;
                        az += pressure_term * gradW.z;
                        
                        // Вязкость (искусственная)
                        float dvx = qvx - vx;
                        float dvy = qvy - vy;
                        float dvz = qvz - vz;
                        float dotp = rx*dvx + ry*dvy + rz*dvz;
                        
                        if (dotp < 0) {
                            // «artificial viscosity»
                            float viscosity_term = -VISCOSITY * qm * dotp 
                                                / (0.1f * SMOOTHING_LENGTH*SMOOTHING_LENGTH + r*r) / qd;
                            ax += viscosity_term * gradW.x;
                            ay += viscosity_term * gradW.y;
                            az += viscosity_term * gradW.z;
                        }
                        
                        // XSPH коррекция скорости
                        float W = cubic_spline_kernel(r, SMOOTHING_LENGTH);
                        vel_avg.x += (qvx - vx) * qm / density_i * W;
                        vel_avg.y += (qvy - vy) * qm / density_i * W;
                        vel_avg.z += (qvz - vz) * qm / density_i * W;
                    }
                }
            }
        }
    }

    // Добавляем XSPH
    const float XSPH_EPS = 0.1f;
    ax += XSPH_EPS * vel_avg.x;
    ay += XSPH_EPS * vel_avg.y;
    az += XSPH_EPS * vel_avg.z;

    // Ограничение ускорения, если нужно
    const float MAX_ACCEL = 1000.0f;
    ax = fminf(fmaxf(ax, -MAX_ACCEL), MAX_ACCEL);
    ay = fminf(fmaxf(ay, -MAX_ACCEL), MAX_ACCEL);
    az = fminf(fmaxf(az, -MAX_ACCEL), MAX_ACCEL);

    accelerations_x[tid] = ax;
    accelerations_y[tid] = ay;
    accelerations_z[tid] = az;
}

/*
 * Интегрирование движения и обработка столкновений
 */
__global__ void integrate_motion(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    float *velocities_x,
    float *velocities_y,
    float *velocities_z,
    float *accelerations_x,
    float *accelerations_y,
    float *accelerations_z,
    int num_particles,
    float time_step
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    // Параметры
    const float DAMPING = 0.4f;         // Сильное демпфирование при отскоке
    const float ENERGY_LOSS = 0.7f;     // Общие потери энергии
    const float MAX_SPEED = 50.0f;      // Ограничение скорости


    // Текущие значения
    float3 pos = {positions_x[tid], positions_y[tid], positions_z[tid]};
    float3 vel = {velocities_x[tid], velocities_y[tid], velocities_z[tid]};
    float3 acc = {accelerations_x[tid], accelerations_y[tid], accelerations_z[tid]};

    // Обновляем скорость с учетом ускорений
    vel.x += acc.x * time_step;
    vel.y += acc.y * time_step;
    vel.z += acc.z * time_step;

    // Глобальное демпфирование скорости
    vel.x *= ENERGY_LOSS;
    vel.y *= ENERGY_LOSS;
    vel.z *= ENERGY_LOSS;

    const float MIN_SPEED = 0.001f;      // Порог зануления скорости
    const float BOUNDARY_FORCE = 40.0f; // Сила возврата частиц
    
    auto handle_boundary = [&](float &pos_comp, float &vel_comp, float boundary) {
        float penetration = 0.0f;
        
        if (pos_comp < -boundary) {
            penetration = (-boundary - pos_comp);
            pos_comp = -boundary + 0.001f;
        }
        else if (pos_comp > boundary) {
            penetration = (pos_comp - boundary);
            pos_comp = boundary - 0.001f;
        }
        
        if (penetration > 0.0f) {
            // Сила сопротивления пропорциональная проникновению
            vel_comp *= -DAMPING;
            vel_comp -= BOUNDARY_FORCE * penetration * time_step;
            
            // Зануление малых скоростей
            if (fabsf(vel_comp) < MIN_SPEED) {
                vel_comp = 0.0f;
            }
        }
    };

    handle_boundary(pos.x, vel.x, BOUNDARY);
    handle_boundary(pos.y, vel.y, BOUNDARY);
    handle_boundary(pos.z, vel.z, BOUNDARY);

    // Ограничение максимальной скорости
    const float speed = sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
    if (speed > MAX_SPEED) {
        const float scale = MAX_SPEED / speed;
        vel.x *= scale;
        vel.y *= scale;
        vel.z *= scale;
    }

    // Обновление позиции после обработки границ
    pos.x += vel.x * time_step;
    pos.y += vel.y * time_step;
    pos.z += vel.z * time_step;

    // Сохранение результатов
    positions_x[tid] = pos.x;
    positions_y[tid] = pos.y;
    positions_z[tid] = pos.z;

    velocities_x[tid] = vel.x;
    velocities_y[tid] = vel.y;
    velocities_z[tid] = vel.z;

    // Сброс ускорений
    accelerations_x[tid] = 0.0f;
    accelerations_y[tid] = 0.0f;
    accelerations_z[tid] = 0.0f;
}

/*
 * Идентификация граничных частиц для обмена между доменами
 */
__global__ void identify_boundary_particles(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    int *boundary_flags,
    int num_particles,
    float boundary_position,
    float boundary_threshold,
    int domain_id
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_particles) return;
    
    float px = positions_x[tid];
    
    // Устанавливаем флаг для частиц на границе
    if (domain_id == 0) {  // Левый домен
        boundary_flags[tid] = (px > boundary_position - boundary_threshold) ? 1 : 0;
    } else {  // Правый домен
        boundary_flags[tid] = (px < boundary_position + boundary_threshold) ? 1 : 0;
    }
}
/*
 * Вычисление вклада граничных частиц в ускорения
 */
__global__ void compute_boundary_acceleration_contribution(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    float *velocities_x,
    float *velocities_y,
    float *velocities_z,
    float *densities,
    float *pressures,
    float *accelerations_x,
    float *accelerations_y,
    float *accelerations_z,
    int num_particles,
    float *boundary_pos_x,
    float *boundary_pos_y,
    float *boundary_pos_z,
    float *boundary_vel_x,
    float *boundary_vel_y,
    float *boundary_vel_z,
    float *boundary_mass,
    float *boundary_density,
    float *boundary_pressure,
    int num_boundary_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;
    
    float px = positions_x[tid];
    float py = positions_y[tid];
    float pz = positions_z[tid];
    float vx = velocities_x[tid];
    float vy = velocities_y[tid];
    float vz = velocities_z[tid];
    float density_i = densities[tid];
    float pressure_i = pressures[tid];
    
    float ax = accelerations_x[tid];
    float ay = accelerations_y[tid];
    float az = accelerations_z[tid];
    
    float3 vel_avg = {0.0f, 0.0f, 0.0f};
    
    // Вычисляем вклад граничных частиц
    for (int j = 0; j < num_boundary_particles; j++) {
        float qx = boundary_pos_x[j];
        float qy = boundary_pos_y[j];
        float qz = boundary_pos_z[j];
        float qvx = boundary_vel_x[j];
        float qvy = boundary_vel_y[j];
        float qvz = boundary_vel_z[j];
        float qm = boundary_mass[j];
        float qd = boundary_density[j];
        float qp = boundary_pressure[j];
        
        // Расстояние и направление к граничной частице
        float rx = px - qx;
        float ry = py - qy;
        float rz = pz - qz;
        float r = sqrtf(rx*rx + ry*ry + rz*rz);
        
        if (r < 2.0f * SMOOTHING_LENGTH && qd > EPSILON) {
            float3 r_vec = make_float3(rx, ry, rz);
            
            // Градиент ядра
            float3 gradW = cubic_spline_kernel_gradient(r_vec, r, SMOOTHING_LENGTH);
            
            // Давление (симметричная формула)
            float pressure_term = -qm * (pressure_i/(density_i*density_i) + qp/(qd*qd));
            ax += pressure_term * gradW.x;
            ay += pressure_term * gradW.y;
            az += pressure_term * gradW.z;
            
            // Вязкость
            float dvx = qvx - vx;
            float dvy = qvy - vy;
            float dvz = qvz - vz;
            float dotp = rx*dvx + ry*dvy + rz*dvz;
            
            if (dotp < 0) {
                // «artificial viscosity»
                float viscosity_term = -VISCOSITY * qm * dotp 
                                    / (0.1f * SMOOTHING_LENGTH*SMOOTHING_LENGTH + r*r) / qd;
                ax += viscosity_term * gradW.x;
                ay += viscosity_term * gradW.y;
                az += viscosity_term * gradW.z;
            }
            
            // XSPH коррекция скорости
            float W = cubic_spline_kernel(r, SMOOTHING_LENGTH);
            vel_avg.x += (qvx - vx) * qm / density_i * W;
            vel_avg.y += (qvy - vy) * qm / density_i * W;
            vel_avg.z += (qvz - vz) * qm / density_i * W;
        }
    }
    
    // Добавляем XSPH вклад от граничных частиц
    const float XSPH_EPS = 0.1f;
    ax += XSPH_EPS * vel_avg.x;
    ay += XSPH_EPS * vel_avg.y;
    az += XSPH_EPS * vel_avg.z;
    
    accelerations_x[tid] = ax;
    accelerations_y[tid] = ay;
    accelerations_z[tid] = az;
}

/*
 * Вычисление вклада граничных частиц в плотность
 */
__global__ void compute_boundary_density_contribution(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    float *densities,
    const int num_particles,
    const float * __restrict__ boundary_pos_x,
    const float * __restrict__ boundary_pos_y,
    const float * __restrict__ boundary_pos_z,
    const float * __restrict__ boundary_mass,
    const int num_boundary
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверка валидности индексов и указателей
    if (tid >= num_particles || num_boundary <= 0) return;
    if (!boundary_pos_x || !boundary_pos_y || !boundary_pos_z || !boundary_mass) return;

    const float px = positions_x[tid];
    const float py = positions_y[tid];
    const float pz = positions_z[tid];
    float density = densities[tid];

    // Каждая основная частица обрабатывает ВСЕ граничные частицы
    for (int j = 0; j < num_boundary; j++) {
        const float qx = boundary_pos_x[j];
        const float qy = boundary_pos_y[j];
        const float qz = boundary_pos_z[j];
        
        const float dx = px - qx;
        const float dy = py - qy;
        const float dz = pz - qz;
        const float r = sqrtf(dx*dx + dy*dy + dz*dz);

        if (r < 2.0f * SMOOTHING_LENGTH) {
            density += boundary_mass[j] * cubic_spline_kernel(r, SMOOTHING_LENGTH);
        }
    }

    densities[tid] = density;
}

"""

class OptimizedSPHSimulation:
    def __init__(self, gpu_id, domain_id, num_particles=1000):
        """
            gpu_id Индекс GPU
            domain_id Индекс домена 0 левый, 1  правый
            num_particles Количество частиц для моделирования
        """
        cuda.init()
        self.device = cuda.Device(gpu_id)
        self.context = self.device.make_context()
        
        self.domain_id = domain_id
        self.num_particles = num_particles
        self.boundary_position = 2.0  # граница между частями
        self.boundary_threshold = 0.1  # Порог для обмена частицами
        self.time_step = 0.005
        self.boundary_particles_ready = False
                
        self.determine_optimal_block_size()
        self.compile_kernels()
        
        # Инициализация памяти и данных
        self.initialize_memory()
        self.initialize_particles()
        
        self.context.pop()
        
        print(f"GPU {gpu_id} SPH for domain {domain_id}")
    
    def determine_optimal_block_size(self):
        device_attrs = self.device.get_attributes()
        max_threads_per_block = device_attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
        self.block_size = 2**int(math.log2(max_threads_per_block))
        
        self.block_size = min(256, self.block_size)
        
        print(f"block size: {self.block_size}")
    
    def compile_kernels(self):
        self.context.push()
        try:
            self.module = SourceModule(cuda_kernel_code)
            
            self.build_hash_grid_kernel = self.module.get_function("build_hash_grid")
            self.compute_density_kernel = self.module.get_function("compute_density")
            self.compute_pressure_kernel = self.module.get_function("compute_pressure")
            self.compute_acceleration_kernel = self.module.get_function("compute_acceleration")
            self.integrate_motion_kernel = self.module.get_function("integrate_motion")
            self.identify_boundary_kernel = self.module.get_function("identify_boundary_particles")
            
            #  для граничных частиц
            self.compute_boundary_density_kernel = self.module.get_function("compute_boundary_density_contribution")
            self.compute_boundary_acceleration_kernel = self.module.get_function("compute_boundary_acceleration_contribution")
        finally:
            self.context.pop()
    
    def initialize_memory(self):
        self.context.push()
        
        try:
            # выделение памяти на GPU
            float_size = np.float32().itemsize
            int_size = np.int32().itemsize
            
            #данные частиц
            self.positions_x_gpu = cuda.mem_alloc(self.num_particles * float_size)
            self.positions_y_gpu = cuda.mem_alloc(self.num_particles * float_size)
            self.positions_z_gpu = cuda.mem_alloc(self.num_particles * float_size)
            
            self.velocities_x_gpu = cuda.mem_alloc(self.num_particles * float_size)
            self.velocities_y_gpu = cuda.mem_alloc(self.num_particles * float_size)
            self.velocities_z_gpu = cuda.mem_alloc(self.num_particles * float_size)
            
            self.masses_gpu = cuda.mem_alloc(self.num_particles * float_size)
            self.densities_gpu = cuda.mem_alloc(self.num_particles * float_size)
            self.pressures_gpu = cuda.mem_alloc(self.num_particles * float_size)
            
            self.accelerations_x_gpu = cuda.mem_alloc(self.num_particles * float_size)
            self.accelerations_y_gpu = cuda.mem_alloc(self.num_particles * float_size)
            self.accelerations_z_gpu = cuda.mem_alloc(self.num_particles * float_size)
            
            #для пространственной сетки поиска соседей
            self.grid_size = 20  
            self.cell_size = 20.0 / self.grid_size # Размер ячейки
            
            grid_total_cells = self.grid_size * self.grid_size * self.grid_size
            
            self.cell_start_gpu = cuda.mem_alloc(grid_total_cells * int_size)
            self.cell_end_gpu = cuda.mem_alloc(grid_total_cells * int_size)
            self.particle_indices_gpu = cuda.mem_alloc(self.num_particles * int_size)
            
            # данные для идентификации граничных частиц
            self.boundary_flags_gpu = cuda.mem_alloc(self.num_particles * int_size)
            
            #для хранения граничных частиц
            self.boundary_particles = []
            self.received_boundary_particles = []
            
            init_cell_start = np.full(grid_total_cells, -1, dtype=np.int32)
            init_cell_end = np.zeros(grid_total_cells, dtype=np.int32)
            cuda.memcpy_htod(self.cell_start_gpu, init_cell_start)
            cuda.memcpy_htod(self.cell_end_gpu, init_cell_end)
        finally:
            self.context.pop()
    
    def initialize_particles(self):
        self.context.push()
        
        try:
            positions_x = np.zeros(self.num_particles, dtype=np.float32)
            positions_y = np.zeros(self.num_particles, dtype=np.float32)
            positions_z = np.zeros(self.num_particles, dtype=np.float32)
            radius = .4
            for i in range(self.num_particles):
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(0, np.pi)
                r = radius * np.cbrt(np.random.uniform(0, 1))
                
                positions_x[i] = r * np.sin(phi) * np.cos(theta)
                positions_y[i] = r * np.sin(phi) * np.sin(theta)
                positions_z[i] = r * np.cos(phi)
            
            velocities_x = np.zeros(self.num_particles, dtype=np.float32)
            velocities_y = np.zeros(self.num_particles, dtype=np.float32)
            velocities_z = np.zeros(self.num_particles, dtype=np.float32)
            
            masses = np.ones(self.num_particles, dtype=np.float32) 
            densities = np.ones(self.num_particles, dtype=np.float32) * 1000.0
            pressures = np.ones(self.num_particles, dtype=np.float32) * 10000.0
      
            # if self.domain_id == 0:  #left
            #     for i in range(self.num_particles):
            #         positions_x[i] = np.random.uniform(-8.0, -2.0)
            #         positions_y[i] = np.random.uniform(-5.0, 5.0)
            #         positions_z[i] = np.random.uniform(-5.0, 5.0)
            # else:  #right
            #     for i in range(self.num_particles):
            #         positions_x[i] = np.random.uniform(2.0, 8.0)
            #         positions_y[i] = np.random.uniform(-5.0, 5.0)
            #         positions_z[i] = np.random.uniform(-5.0, 5.0)
            
            #копируем на GPU
            cuda.memcpy_htod(self.positions_x_gpu, positions_x)
            cuda.memcpy_htod(self.positions_y_gpu, positions_y)
            cuda.memcpy_htod(self.positions_z_gpu, positions_z)
            
            cuda.memcpy_htod(self.velocities_x_gpu, velocities_x)
            cuda.memcpy_htod(self.velocities_y_gpu, velocities_y)
            cuda.memcpy_htod(self.velocities_z_gpu, velocities_z)
            
            cuda.memcpy_htod(self.masses_gpu, masses)
            cuda.memcpy_htod(self.densities_gpu, densities)
            cuda.memcpy_htod(self.pressures_gpu, pressures)
            
            #  копии на cpu данных в CPU для отладки и визуализации
            self.positions_x = positions_x
            self.positions_y = positions_y
            self.positions_z = positions_z
            self.velocities_x = velocities_x
            self.velocities_y = velocities_y
            self.velocities_z = velocities_z
            self.masses = masses
            self.densities = densities
            self.pressures = pressures
        finally:
            self.context.pop()
    
    def simulate_step(self, time_step=None):
        if time_step is not None:
            self.time_step = time_step
        
        self.context.push()
        
        try:
            #расчет количества блоков в зависимости от частиц
            grid_size = (self.num_particles + self.block_size - 1) // self.block_size
            
            # очистка данных хешсети
            init_cell_start = np.full(self.grid_size**3, -1, dtype=np.int32)
            init_cell_end = np.zeros(self.grid_size**3, dtype=np.int32)
            
            cuda.memcpy_htod(self.cell_start_gpu, init_cell_start)
            cuda.memcpy_htod(self.cell_end_gpu, init_cell_end)
            
            # построение хеш-сетки для эффективного поиска соседей
            try:
                self.build_hash_grid_kernel(
                    self.cell_start_gpu,
                    self.cell_end_gpu,
                    self.particle_indices_gpu,
                    self.positions_x_gpu,
                    self.positions_y_gpu,
                    self.positions_z_gpu,
                    np.int32(self.num_particles),
                    np.float32(self.cell_size),
                    np.int32(self.grid_size),
                    block=(self.block_size, 1, 1),
                    grid=(grid_size, 1)
                )
                self.context.synchronize() 
            except Exception as e:
                print(f"build_hash_grid_kernel")
            
            # вычисление плотности 
            shared_memory_size = self.block_size * 4 * 4  # 4 float для каждой x,y,z,mass
            try:
                self.compute_density_kernel(
                    self.positions_x_gpu,
                    self.positions_y_gpu,
                    self.positions_z_gpu,
                    self.masses_gpu,
                    self.densities_gpu,
                    self.cell_start_gpu,
                    self.cell_end_gpu,
                    self.particle_indices_gpu,
                    np.int32(self.num_particles),
                    np.float32(self.cell_size),
                    np.int32(self.grid_size),
                    block=(self.block_size, 1, 1),
                    grid=(grid_size, 1),
                    shared=shared_memory_size
                )
                self.context.synchronize() 
            except Exception as e:
                print(f"compute_density_kernel")
            # вычисление плотности от граничных частиц
            if hasattr(self, 'boundary_particles_ready') and self.boundary_particles_ready and self.num_boundary_particles > 0:
                try:
                    self.compute_boundary_density_kernel(
                        self.positions_x_gpu,
                        self.positions_y_gpu,
                        self.positions_z_gpu,
                        self.densities_gpu,
                        np.int32(self.num_particles),
                        self.boundary_pos_x_gpu,
                        self.boundary_pos_y_gpu,
                        self.boundary_pos_z_gpu,
                        self.boundary_mass_gpu,
                        np.int32(self.num_boundary_particles),
                        block=(self.block_size, 1, 1),
                        grid=(grid_size, 1)
                    )
                    self.context.synchronize() 
                except Exception as e:
                    print(f"compute_boundary_density_kernel")
            # вычисление давления из плотности
            try:
                self.compute_pressure_kernel(
                    self.densities_gpu,
                    self.pressures_gpu,
                    np.int32(self.num_particles),
                    block=(self.block_size, 1, 1),
                    grid=(grid_size, 1)
                )
                self.context.synchronize() 
            except Exception as e:
                    print(f"compute_pressure_kernel")
            #вычисление сил и ускорений
            shared_memory_size = self.block_size * 9 * 4  # 9 float-значений для каждой частицы
            try:
                self.compute_acceleration_kernel(
                    self.positions_x_gpu,
                    self.positions_y_gpu,
                    self.positions_z_gpu,
                    self.velocities_x_gpu,
                    self.velocities_y_gpu,
                    self.velocities_z_gpu,
                    self.masses_gpu,
                    self.densities_gpu,
                    self.pressures_gpu,
                    self.accelerations_x_gpu,
                    self.accelerations_y_gpu,
                    self.accelerations_z_gpu,
                    self.cell_start_gpu,
                    self.cell_end_gpu,
                    self.particle_indices_gpu,
                    np.int32(self.num_particles),
                    np.float32(self.cell_size),
                    np.int32(self.grid_size),
                    block=(self.block_size, 1, 1),
                    grid=(grid_size, 1),
                    shared=shared_memory_size
                )
                self.context.synchronize() 
            except Exception as e:
                print(f"compute_acceleration_kernel")
                
            # вычисление ускорения от граничных яастиц
            if hasattr(self, 'boundary_particles_ready') and self.boundary_particles_ready and self.num_boundary_particles > 0:
                try:
                    self.compute_boundary_acceleration_kernel(
                        self.positions_x_gpu,
                        self.positions_y_gpu,
                        self.positions_z_gpu,
                        self.velocities_x_gpu,
                        self.velocities_y_gpu,
                        self.velocities_z_gpu,
                        self.densities_gpu,
                        self.pressures_gpu,
                        self.accelerations_x_gpu,
                        self.accelerations_y_gpu,
                        self.accelerations_z_gpu,
                        np.int32(self.num_particles),
                        self.boundary_pos_x_gpu,
                        self.boundary_pos_y_gpu,
                        self.boundary_pos_z_gpu,
                        self.boundary_vel_x_gpu,
                        self.boundary_vel_y_gpu,
                        self.boundary_vel_z_gpu,
                        self.boundary_mass_gpu,
                        self.boundary_density_gpu,
                        self.boundary_pressure_gpu,
                        np.int32(self.num_boundary_particles),
                        block=(self.block_size, 1, 1),
                        grid=(grid_size, 1)
                    )
                except Exception as e:
                    print(f"compute_boundary_acceleration_kernel")
            #  движение
            try:
                self.integrate_motion_kernel(
                    self.positions_x_gpu,
                    self.positions_y_gpu,
                    self.positions_z_gpu,
                    self.velocities_x_gpu,
                    self.velocities_y_gpu,
                    self.velocities_z_gpu,
                    self.accelerations_x_gpu,
                    self.accelerations_y_gpu,
                    self.accelerations_z_gpu,
                    np.int32(self.num_particles),
                    np.float32(self.time_step),
                    block=(self.block_size, 1, 1),
                    grid=(grid_size, 1)
                )
                self.context.synchronize() 
            except Exception as e:
                print(f"integrate_motion_kernel")
            # вычисление граничных частиц для обмена
            try:
                self.identify_boundary_kernel(
                    self.positions_x_gpu,
                    self.positions_y_gpu,
                    self.positions_z_gpu,
                    self.boundary_flags_gpu,
                    np.int32(self.num_particles),
                    np.float32(self.boundary_position),
                    np.float32(self.boundary_threshold),
                    np.int32(self.domain_id),
                    block=(self.block_size, 1, 1),
                    grid=(grid_size, 1)
                )
                self.context.synchronize() 
            except Exception as e:
                print(f"identify_boundary_kernel")
            #копируем данные обратно в CPU для отправки через grpc
            cuda.memcpy_dtoh(self.positions_x, self.positions_x_gpu)
            cuda.memcpy_dtoh(self.positions_y, self.positions_y_gpu)
            cuda.memcpy_dtoh(self.positions_z, self.positions_z_gpu)
            cuda.memcpy_dtoh(self.velocities_x, self.velocities_x_gpu)
            cuda.memcpy_dtoh(self.velocities_y, self.velocities_y_gpu)
            cuda.memcpy_dtoh(self.velocities_z, self.velocities_z_gpu)
            cuda.memcpy_dtoh(self.densities, self.densities_gpu)
            cuda.memcpy_dtoh(self.pressures, self.pressures_gpu)
            
            #флаги граничных частиц
            boundary_flags = np.zeros(self.num_particles, dtype=np.int32)
            cuda.memcpy_dtoh(boundary_flags, self.boundary_flags_gpu)
            
            #список граничных частиц
            self.boundary_particles = []
            for i in range(self.num_particles):
                if boundary_flags[i] == 1:
                    self.boundary_particles.append((
                        self.positions_x[i],
                        self.positions_y[i],
                        self.positions_z[i],
                        self.velocities_x[i],
                        self.velocities_y[i],
                        self.velocities_z[i],
                        self.masses[i],
                        self.densities[i],
                        self.pressures[i]
                    ))
        finally:
            self.context.pop()
        
        return len(self.boundary_particles)
    
    def add_boundary_particles(self, particles):
        """
            particles - (x, y, z, vx, vy, vz, mass, density, pressure)
        """
        self.received_boundary_particles = particles
    
    def process_boundary_particles(self):
        if not self.received_boundary_particles:
            return
        
        self.context.push()
        
        try:
            self.boundary_particles_ready = False
            
            num_boundary = len(self.received_boundary_particles)
            if num_boundary == 0:
                return
                
         
            b_pos_x = np.array([p[0] for p in self.received_boundary_particles], dtype=np.float32)
            b_pos_y = np.array([p[1] for p in self.received_boundary_particles], dtype=np.float32)
            b_pos_z = np.array([p[2] for p in self.received_boundary_particles], dtype=np.float32)
            b_vel_x = np.array([p[3] for p in self.received_boundary_particles], dtype=np.float32)
            b_vel_y = np.array([p[4] for p in self.received_boundary_particles], dtype=np.float32)
            b_vel_z = np.array([p[5] for p in self.received_boundary_particles], dtype=np.float32)
            b_mass = np.array([p[6] for p in self.received_boundary_particles], dtype=np.float32)
            b_density = np.array([p[7] for p in self.received_boundary_particles], dtype=np.float32)
            b_pressure = np.array([p[8] for p in self.received_boundary_particles], dtype=np.float32)
            
            # Проверка данных на корректность
            if np.isnan(b_pos_x).any() or np.isnan(b_vel_x).any() or np.isnan(b_mass).any():
                return
                
            try:
                # выделяем буферы для граничных частиц, включая скорости
                if not hasattr(self, 'boundary_pos_x_gpu') or not self.boundary_pos_x_gpu:
                    self.boundary_pos_x_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_pos_y_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_pos_z_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_vel_x_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_vel_y_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_vel_z_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_mass_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_density_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_pressure_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_buffer_size = num_boundary
                elif self.boundary_buffer_size < num_boundary:
                    # освобождаем старые буферы
                    self.boundary_pos_x_gpu.free()
                    self.boundary_pos_y_gpu.free()
                    self.boundary_pos_z_gpu.free()
                    self.boundary_vel_x_gpu.free()
                    self.boundary_vel_y_gpu.free()
                    self.boundary_vel_z_gpu.free()
                    self.boundary_mass_gpu.free()
                    
                   
                    self.boundary_pos_x_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_pos_y_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_pos_z_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_vel_x_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_vel_y_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_vel_z_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_mass_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_density_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_pressure_gpu = cuda.mem_alloc(num_boundary * np.float32().itemsize)
                    self.boundary_buffer_size = num_boundary
                    
                cuda.memcpy_htod(self.boundary_pos_x_gpu, b_pos_x)
                cuda.memcpy_htod(self.boundary_pos_y_gpu, b_pos_y)
                cuda.memcpy_htod(self.boundary_pos_z_gpu, b_pos_z)
                cuda.memcpy_htod(self.boundary_vel_x_gpu, b_vel_x)
                cuda.memcpy_htod(self.boundary_vel_y_gpu, b_vel_y)
                cuda.memcpy_htod(self.boundary_vel_z_gpu, b_vel_z)
                cuda.memcpy_htod(self.boundary_mass_gpu, b_mass)
                cuda.memcpy_htod(self.boundary_density_gpu, b_density)
                cuda.memcpy_htod(self.boundary_pressure_gpu, b_pressure)
                
                self.num_boundary_particles = num_boundary
                self.boundary_particles_ready = True
                
                print(f"Boundary particles ready: {num_boundary}")
            except cuda.LogicError as e:
                print(f"CUDA error boundary particles : {e}")
                self.boundary_particles_ready = False
            
        except Exception as e:
            print(f"Exception oundary particles ready: {e}")
            self.boundary_particles_ready = False
        finally:
            self.received_boundary_particles = []
            self.context.pop()


    
    def get_particle_data(self):
        return {
            'positions_x': self.positions_x.copy(),
            'positions_y': self.positions_y.copy(),
            'positions_z': self.positions_z.copy(),
            'velocities_x': self.velocities_x.copy(),
            'velocities_y': self.velocities_y.copy(),
            'velocities_z': self.velocities_z.copy(),
            'masses': self.masses.copy(),
            'densities': self.densities.copy(),
            'pressures': self.pressures.copy()
        }
    
    def get_boundary_particles(self):
        return self.boundary_particles.copy()
    
    def cleanup(self):
        self.context.push()
        
        try:
            self.positions_x_gpu.free()
            self.positions_y_gpu.free()
            self.positions_z_gpu.free()
            
            self.velocities_x_gpu.free()
            self.velocities_y_gpu.free()
            self.velocities_z_gpu.free()
            
            self.masses_gpu.free()
            self.densities_gpu.free()
            self.pressures_gpu.free()
            
            self.accelerations_x_gpu.free()
            self.accelerations_y_gpu.free()
            self.accelerations_z_gpu.free()
            
            self.cell_start_gpu.free()
            self.cell_end_gpu.free()
            self.particle_indices_gpu.free()
            
            self.boundary_flags_gpu.free()
            
            self.boundary_pos_x_gpu.free()
            self.boundary_pos_y_gpu.free()
            self.boundary_pos_z_gpu.free()
            self.boundary_vel_x_gpu.free()
            self.boundary_vel_y_gpu.free()
            self.boundary_vel_z_gpu.free()
            self.boundary_mass_gpu.free()
            self.boundary_density_gpu.free()
            self.boundary_pressure_gpu.free()
        finally:
            self.context.pop()
            self.context.detach()


class OptimizedSPHServer(sph_optimized_pb2_grpc.OptimizedSPHSimulationServicer):
    def __init__(self, gpu_id, domain_id, num_particles=1000):
        self.gpu_id = gpu_id
        self.domain_id = domain_id
        self.simulation = OptimizedSPHSimulation(gpu_id, domain_id, num_particles)
        self.logger = SimulationLogger(log_dir=f"logs/domain_{domain_id}")
        self.lock = threading.Lock()
        print(f"SPH Server initialized on GPU {gpu_id} for domain {domain_id}")
    
    def SimulateStep(self, request, context):
        with self.lock:
            log_data = {
                    'time_step': self.simulation.time_step,
                    'positions_x': self.simulation.positions_x,
                    'positions_y': self.simulation.positions_y,
                    'positions_z': self.simulation.positions_z,
                    'velocities_x': self.simulation.velocities_x,
                    'velocities_y': self.simulation.velocities_y,
                    'velocities_z': self.simulation.velocities_z,
                    'densities': self.simulation.densities,
                    'pressures': self.simulation.pressures
                }
                
            self.logger.log_step(log_data)
            time_step = request.time_step
            self.simulation.boundary_position = request.boundary_position
            
            self.simulation.process_boundary_particles()
            
            num_boundary = self.simulation.simulate_step(time_step)
            print(f"Domain {self.domain_id} on GPU {self.gpu_id} simulated step with {num_boundary} boundary particles")
            
            particle_data = self.simulation.get_particle_data()
            
            response = sph_optimized_pb2.SimulationResponse(
                positions_x=particle_data['positions_x'],
                positions_y=particle_data['positions_y'],
                positions_z=particle_data['positions_z'],
                velocities_x=particle_data['velocities_x'],
                velocities_y=particle_data['velocities_y'],
                velocities_z=particle_data['velocities_z'],
                densities=particle_data['densities'],
                pressures=particle_data['pressures']
            )
            
            return response
    
    def ExchangeBoundaryParticles(self, request, context):

        with self.lock:
            boundary_particles = []
            
            for i in range(len(request.positions_x)):
                boundary_particles.append((
                    request.positions_x[i],
                    request.positions_y[i],
                    request.positions_z[i],
                    request.velocities_x[i],
                    request.velocities_y[i],
                    request.velocities_z[i],
                    request.masses[i],
                    request.densities[i],
                    request.pressures[i]
                ))
            
            self.simulation.add_boundary_particles(boundary_particles)
            
            print(f"Domain {self.domain_id} on GPU {self.gpu_id} received {len(boundary_particles)} boundary particles")
            
            return sph_optimized_pb2.BoundaryResponse(success=True)
    
    def cleanup(self):
        self.simulation.cleanup()

def serve(gpu_id, domain_id, num_particles=1000):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sph_server = OptimizedSPHServer(gpu_id, domain_id, num_particles)
    sph_optimized_pb2_grpc.add_OptimizedSPHSimulationServicer_to_server(
        sph_server, server)
    
    port = 50051 + gpu_id
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"SPH Server for GPU {gpu_id}, domain {domain_id} started on port {port}")
    
    try:
        while True:
            time.sleep(86400)  #  цикл
    except KeyboardInterrupt:
        sph_server.cleanup()
        server.stop(0)
        
        
class SimulationLogger:
    def __init__(self, log_dir="logs", max_log_size=10*1024*1024, backup_count=5):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("SPHLogger")
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Ротация логов
        file_handler = RotatingFileHandler(
            self.log_dir / 'sph_simulation.log',
            maxBytes=max_log_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.stats = {
            'step_count': 0,
            'nan_detected': False,
            'energy_history': [],
            'density_stats': []
        }

    def log_step(self, sim_data):
        self.stats['step_count'] += 1
        
        self._check_for_nan(sim_data)
        self._check_energy(sim_data)
        self._log_basic_stats(sim_data)
        self._log_particle_stats(sim_data)
        
        if self.stats['step_count'] % 100 == 0:
            self._save_snapshot(sim_data)

    def _check_for_nan(self, data):
        has_nan = any(
            np.isnan(data[key]).any()
            for key in ['positions_x', 'positions_y', 'positions_z',
                       'velocities_x', 'velocities_y', 'velocities_z',
                       'densities', 'pressures']
        )
        
        if has_nan and not self.stats['nan_detected']:
            self.logger.error("Обнаружены NaN значения в данных симуляции!")
            self.stats['nan_detected'] = True
            self._save_debug_data(data)

    def _check_energy(self, data):
        kinetic = np.sum(data['velocities_x']**2 + 
                      data['velocities_y']**2 + 
                      data['velocities_z']**2)
        
        potential = np.sum(data['positions_z'] * 9.81)
        
        energy_change = abs(kinetic + potential - self.stats.get('last_energy', 0))
        self.stats['last_energy'] = kinetic + potential
        self.stats['energy_history'].append(kinetic + potential)
        
        if energy_change > 1e5:
            self.logger.warning(f"Резкое изменение энергии: {energy_change:.2f} Дж")

    def _log_basic_stats(self, data):
        stats_msg = (
            f"Шаг {self.stats['step_count']}:\n"
            f"  • Временной шаг: {data['time_step']:.6f} сек\n"
            f"  • Частиц: {len(data['positions_x'])}\n"
            f"  • Плотность: μ={np.mean(data['densities']):.1f} σ={np.std(data['densities']):.1f}\n"
            f"  • Давление: min={np.min(data['pressures']):.1f} max={np.max(data['pressures']):.1f}\n"
            f"  • Скорости: max={np.max(np.abs(data['velocities_x'])):.2f} м/с"
        )
        self.logger.info(stats_msg)

    def _log_particle_stats(self, data):
        outlier_mask = (
            (data['densities'] < 400) | 
            (data['densities'] > 2500) |
            (np.abs(data['velocities_x']) > 150.0)
        )
        
        if np.any(outlier_mask):
            outlier_count = np.sum(outlier_mask)
            self.logger.warning(
                f"Обнаружены аномальные частицы ({outlier_count} шт):\n"
                f"  • Плотность: {data['densities'][outlier_mask][:5]}\n"
                f"  • Скорость X: {data['velocities_x'][outlier_mask][:5]}"
            )

    def _save_snapshot(self, data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"snapshot_{timestamp}.npz"
        
        np.savez_compressed(
            filename,
            positions_x=data['positions_x'],
            positions_y=data['positions_y'],
            positions_z=data['positions_z'],
            velocities=data['velocities_x'],
            densities=data['densities'],
            pressures=data['pressures']
        )
        self.logger.info(f"Сохранен снимок состояния: {filename}")

    def _save_debug_data(self, data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"debug_{timestamp}.npz"
        
        np.savez_compressed(
            filename,
            **data
        )
        self.logger.error(f"Сохранены отладочные данные: {filename}")