#!/bin/awk -f

BEGIN {
  FS = ","
}

{
  tipo = $85
  
  # Elimina las comillas de principio y fin del campo 85
  gsub("\"", "", tipo)
  
  # Crea un archivo de salida basado en el tipo del campo 85
  archivo_salida = tipo ".csv"
  
  # Añade la línea actual al archivo de salida correspondiente
  print >> archivo_salida
}

END {
  # Cierra todos los archivos de salida
  for (archivo in archivos_salida) {
    close(archivo)
  }
}