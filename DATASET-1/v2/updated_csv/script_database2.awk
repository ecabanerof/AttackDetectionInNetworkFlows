#!/usr/bin/awk -f

BEGIN {
    FS = ",";
    OFS = ",";

    # Define the category mapping
    category_mapping["BENIGN"] = "BENIGN";
    category_mapping["Bruteforce DNS"] = "Bruteforce";
    category_mapping["Bruteforce FTP"] = "Bruteforce";
    category_mapping["Bruteforce HTTP"] = "Bruteforce";
    category_mapping["Bruteforce SSH"] = "Bruteforce";
    category_mapping["Bruteforce Telnet"] = "Bruteforce";
    category_mapping["DoS ACK"] = "DoS";
    category_mapping["DoS CWR"] = "DoS";
    category_mapping["DoS ECN"] = "DoS";
    category_mapping["DoS FIN"] = "DoS";
    category_mapping["DoS HTTP"] = "DoS";
    category_mapping["DoS ICMP"] = "DoS";
    category_mapping["DoS MAC"] = "DoS";
    category_mapping["DoS PSH"] = "DoS";
    category_mapping["DoS RST"] = "DoS";
    category_mapping["DoS SYN"] = "DoS";
    category_mapping["DoS UDP"] = "DoS";
    category_mapping["DoS URG"] = "DoS";
    category_mapping["Information Gathering"] = "Information Gathering";
    category_mapping["Mirai DDoS ACK"] = "Mirai";
    category_mapping["Mirai DDoS DNS"] = "Mirai";
    category_mapping["Mirai DDoS GREETH"] = "Mirai";
    category_mapping["Mirai DDoS GREIP"] = "Mirai";
    category_mapping["Mirai DDoS HTTP"] = "Mirai";
    category_mapping["Mirai DDoS SYN"] = "Mirai";
    category_mapping["Mirai DDoS UDP"] = "Mirai";
    category_mapping["Mirai Scan Bruteforce"] = "Mirai";

    # Initialize category counts
    for (category in category_mapping) {
        category_counts[category_mapping[category]] = 0;
    }

    # Define the maximum number of samples per category
    max_samples["DoS"] = 200000;
    max_samples["Information Gathering"] = 150000;
    max_samples["Mirai"] = 100000;
    max_samples["Bruteforce"] = 100000;

    # Initialize line counter
    line_counter = 0;
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
    for (i = 1; i <= NF; i++) {
        if (i != 1 && i != 2 && i != 3 && i != 4 && i != 5 && i != 6 && i != 8 && i != 86 && i != 87 && i != 88 && i != 89) {
            header[++header_count] = $i;
        }
    }

    for (i = 1; i <= header_count; i++) {
        printf "%s%s", header[i], (i < header_count ? OFS : ORS);
    }

    next;
}

{
    # Increment line counter
    line_counter++;
    if (line_counter % 100000 == 0) {
        print "Processing line:", line_counter > "/dev/stderr";
    }

    # Verificar si la fila contiene valores inválidos
    if (!contains_invalid_value($0)) {
        data_count = 0;

        # Procesar los datos
        for (i = 1; i <= NF; i++) {
            if (i != 1 && i != 2 && i != 3 && i != 4 && i != 5 && i != 6 && i != 8 && i != 86 && i != 87 && i != 88 && i != 89) {
                data[++data_count] = $i;
            }
        }

        # Aplicar el mapeo de categorías a la columna 77
        if (data[77] in category_mapping) {
            data[77] = category_mapping[data[77]];
        } else {
            print "Unknown category:", data[77] > "/dev/stderr";
        }

        # Incrementar el conteo de la categoría
        category = data[77];
        if (category_counts[category] < max_samples[category]) {
            category_counts[category]++;

            # Imprimir los datos modificados
            for (i = 1; i <= data_count; i++) {
                printf "%s%s", data[i], (i < data_count ? OFS : ORS);
            }
        }
    } else {
        print "Invalid value found in line:", line_counter > "/dev/stderr";
    }

    # Reiniciar el arreglo de datos para la siguiente fila
    delete data;
    data_count = 0;
}