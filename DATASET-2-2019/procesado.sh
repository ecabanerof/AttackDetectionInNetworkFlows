#!/bin/bash

files=(
    "DrDoS_DNS.csv"
    "DrDoS_LDAP.csv"
    "DrDoS_MSSQL.csv"
    "DrDoS_NetBIOS.csv"
    "DrDoS_NTP.csv"
    "DrDoS_SNMP.csv"
    "DrDoS_SSDP.csv"
    "DrDoS_UDP.csv"
    "LDAP.csv"
    "MSSQL.csv"
    "NetBIOS.csv"
    "Portmap.csv"
    "Syn.csv"
    "Syn2.csv"
    "TFTP.csv"
    "UDP.csv"
    "UDPLag.csv"
    "UDPLag2.csv"
)

total_files=${#files[@]}

for ((i = 0; i < ${#files[@]}; i++)); do
    file="${files[i]}"
    echo "Processing ${file} ($((i + 1)) of ${total_files})..."

    # Verificar si el archivo existe
    if [ ! -f "${file}" ]; then
        echo "Error: File '${file}' not found."
    else
        # Ejecutar el script awk y redirigir la salida y el error a archivos
        awk -F, -f script.awk "${file}" > "processed_${file}" 2> "awk_error_${file}.log"

        # Verificar si el archivo de salida está vacío
        if [ -s "processed_${file}" ]; then
            echo "Processed file 'processed_${file}' created."
        else
            echo "Warning: Processed file 'processed_${file}' is empty."
        fi

        # Verificar si hubo errores en la ejecución de awk
        if [ -s "awk_error_${file}.log" ]; then
            echo "Error: awk errors found in 'awk_error_${file}.log'."
        else
            echo "No awk errors found."
        fi
    fi
done

echo "All files have been processed."