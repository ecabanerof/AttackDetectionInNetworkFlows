#!/bin/bash

# Nombre del archivo a procesar
input_file="data.csv"

# Nombre del archivo de salida
output_file="processed_data.csv"

# Nombre del archivo de log para errores de awk
error_log="awk_error.log"

echo "Processing ${input_file}..."

# Verificar si el archivo existe
if [ ! -f "${input_file}" ]; then
    echo "Error: File '${input_file}' not found."
    exit 1
fi

# Contar el número total de líneas en el archivo de entrada
total_lines=$(wc -l < "${input_file}")
echo "Total lines to process: ${total_lines}"

# Ejecutar el script awk con un contador de progreso
awk -F, -f script_database2 "${input_file}" 2>"${error_log}" | 
    tee "${output_file}" | 
    awk -v total="$total_lines" '
        BEGIN {ORS=""}
        {
            if (NR % 1000 == 0 || NR == total) {
                percent = NR / total * 100
                printf "\rProgress: %d%%", percent
            }
        }
        END {print "\n"}
    '

# Verificar si el archivo de salida está vacío
if [ -s "${output_file}" ]; then
    echo "Processed file '${output_file}' created."
else
    echo "Warning: Processed file '${output_file}' is empty."
fi

# Verificar si hubo errores en la ejecución de awk
if [ -s "${error_log}" ]; then
    echo "Error: awk errors found in '${error_log}'."
else
    echo "No awk errors found."
    # Eliminar el archivo de log si está vacío
    rm "${error_log}"
fi

echo "Processing complete."