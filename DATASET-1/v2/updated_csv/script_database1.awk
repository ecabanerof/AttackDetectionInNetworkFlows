#!/usr/bin/awk -f

BEGIN {
    FS = ",";
    OFS = ",";
}

function contains_invalid_value(line) {
    split(line, arr, FS);
    for (i = 1; i <= length(arr); i++) {
        if (arr[i] ~ /NaN|inf|-inf/) {
            return 1;
        }
    }
    return 0;
}

FNR == 1 {
    # Procesar la cabecera
    for (i = 1; i <= NF; i++) {
        if (i != 1 && i != 2 && i != 3 && i != 4 && i != 5 && i != 6 && i != 8 && i != 86 && i != 87 && i != 88 && i != 89) {
            header[++header_count] = $i;
        }
    }

    # Imprimir la cabecera modificada
    for (i = 1; i <= header_count; i++) {
        printf "%s%s", header[i], (i < header_count ? OFS : ORS);
    }

    next;
}

{
    # Verificar si la fila contiene valores inválidos
    if (!contains_invalid_value($0)) {
        data_count = 0;

        # Procesar los datos
        for (i = 1; i <= NF; i++) {
            if (i != 1 && i != 2 && i != 3 && i != 4 && i != 5 && i != 6 && i != 8 && i != 86 && i != 87 && i != 88 && i != 89) {
                data[++data_count] = $i;
            }
        }

        # Imprimir los datos modificados
        for (i = 1; i <= data_count; i++) {
            printf "%s%s", data[i], (i < data_count ? OFS : ORS);
        }
    }

    # Reiniciar el arreglo de datos para la siguiente fila
    delete data;
    data_count = 0;
}