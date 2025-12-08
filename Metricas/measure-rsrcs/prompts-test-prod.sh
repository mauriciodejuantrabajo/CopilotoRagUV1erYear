#!/bin/bash


#MODELS=("qwen3-coder:30b")
#PROMPTS=("¿Cuándo adquiero la condición de alumno regular y por cuánto tiempo dura la matrícula antes de renovarla?")
MODELS=("qwen3-coder:30b" "gemma3:27b")
TEMPS=(0.0 0.2)
PROMPTS=(
  "¿Cuándo adquiero la condición de alumno regular y por cuánto tiempo dura la matrícula antes de renovarla?"
  "¿Qué diferencia hay entre el sistema regular y los sistemas especiales de ingreso?"
  "Si ingresé con estudios previos, ¿puedo homologar asignaturas y bajo qué condiciones generales?"
  "¿Cuál es la diferencia entre ‘programa’ y ‘carrera’ y cómo se organizan los estudios?"
  "¿Se exige asistencia mínima a todas las asignaturas?"
  "¿Cuál es la escala de notas y con qué nota final se aprueba una asignatura?"
  "Si falto a una evaluación, ¿qué nota obtengo y en qué plazo debo justificar la inasistencia para evitar el 1,0?"
  "¿Cuántas veces puedo repetir un examen final reprobado y qué ocurre si no asisto al examen?"
  "En currículos rígidos y flexibles, ¿cómo opera la promoción y las previaturas para inscribir asignaturas superiores?"
  "¿Cuántas veces puedo cursar una asignatura reprobada y qué ocurre si necesito más oportunidades?"
  "¿Quién define el plazo máximo para terminar la carrera y qué pasa al sobrepasarlo?"
  "¿Cuál es la diferencia entre renuncia, suspensión, postergación y reincorporación, y quién resuelve estas solicitudes?"
  "¿Quién otorga los diplomas y certificados de grado o título y qué documentos integran el expediente para titularse?"
  "¿Cómo se expresa la calificación del título o grado (aprobado, con distinción, con distinción máxima)?"
  "¿Qué significan homologación, transferencia y egresado según el reglamento general?"
  "¿Cuál es el régimen de estudios de ingniería civil informática y qué grados/título otorga la carrera?"
  "¿Cuándo adquiero la condición de egresado en ingniería civil informática?"
  "¿Las previaturas son obligatorias y quién puede autorizar excepciones?"
  "¿Cuál es la duración de la carrera de ingniería civil informática y cuál es el plazo máximo para completarla antes de revalidar?"
  "¿Cuándo debo inscribir asignaturas y qué ocurre si repruebo una?"
  "¿Puedo inscribir asignaturas con tope de horario? ¿Cuál es la carga máxima (SCT) y cuánta carga adicional puedo solicitar?"
  "¿En qué casos el Jefe de Carrera puede eximirme de reglas como tope, reprobación o carga?"
  "¿Qué debe informar el profesor al inicio del curso y quiénes pueden tomar evaluaciones sumativas?"
  "¿Cuál es la asistencia mínima exigible en talleres y laboratorios en ingniería civil informática?"
  "¿Cuál es la escala de notas y qué nivel de exigencia equivale a nota 4,0 en ingniería civil informática?"
  "¿Cuántas evaluaciones sumativas como mínimo debe contemplar una asignatura?"
  "Si falto a una evaluación, ¿qué nota obtengo y en qué plazo debo justificar? ¿Existen evaluaciones recuperativas?"
  "Si termino el curso con nota entre 3,5 y 3,9 inclusive, ¿cómo se calcula la nota final tras la sumativa extraordinaria?"
  "¿En qué plazo deben publicarse las notas y en cuántos días puedo apelar una calificación?"
  "¿Cuándo puedo realizar las Prácticas I y II, y cómo se configura el Trabajo de Título y su comisión evaluadora?"
)


SERVER_SCRIPT=agent_rag_server.py
CLIENT_SCRIPT=agent_rag_client.py
SERVER_LOG_FILE=rag_server.log
RESPONSE_FOLDER=response_results
MEASURE_FOLDER=measure_results
USAGE_TEMP_FILE=usage-temp.txt
RESULT_TEMP_FILE=result-temp.txt
DELIMITER_USAGE=" | "
DELIMITER="~"

# Create directories
mkdir -p "$RESPONSE_FOLDER" "$MEASURE_FOLDER"
rm $RESPONSE_FOLDER/*
rm $MEASURE_FOLDER/*


calculate_metrics() {
  # Init
  entries=0
  total_cpu=0
  max_ram_util=0
  max_ram_mem=0
  max_gpu_util=0
  max_gpu_mem=0
  max_gpu_temp=0
  all_usage="$1"

  # Read each data line stored in 'all_usage'
  while IFS='|' read -r cpu_usage ram_usage gpu_usage; do
    # Clean all spaces
    cpu_usage=$(echo $cpu_usage)
    ram_usage=$(echo $ram_usage)
    gpu_usage=$(echo $gpu_usage)

    # Extract RAM values
    ram_util=$(echo "$ram_usage" | cut -d' ' -f1) # Utilization (%)
    ram_mem=$(echo "$ram_usage" | cut -d' ' -f2)  # Quantity (MB)

    # Extract GPU values
    gpu_util=$(echo "$gpu_usage" | cut -d' ' -f1) # Utilization(%)
    gpu_mem=$(echo "$gpu_usage" | cut -d' ' -f2)  # Quantity (MB)
    gpu_temp=$(echo "$gpu_usage" | cut -d' ' -f3) # Temp (°C)

    # Adding for Average CPU Calculation
    total_cpu=$(echo "$total_cpu + $cpu_usage" | bc)
    entries=$((entries + 1))

    # Find the maximum value of % memory usage
    if [ "$(echo "$ram_util > $max_ram_util" | bc)" -eq 1 ]; then
      max_ram_util=$ram_util
    fi

    # Find the maximum memory used value in MB
    if [ "$ram_mem" -gt "$max_ram_mem" ]; then
      max_ram_mem=$ram_mem
    fi

    # Find maximum GPU values (utilization, memory, and temperature)
    if [ "$gpu_util" -gt "$max_gpu_util" ]; then
      max_gpu_util=$gpu_util
    fi
    if [ "$gpu_mem" -gt "$max_gpu_mem" ]; then
      max_gpu_mem=$gpu_mem
    fi
    if [ "$gpu_temp" -gt "$max_gpu_temp" ]; then
      max_gpu_temp=$gpu_temp
    fi
  done <<< "$all_usage"  # Reading

  # Calculate avg CPU
  avg_cpu=$(echo "scale=2; $total_cpu / $entries" | bc) # 2 decimals

  echo "CPU avg (%): $avg_cpu | Mem (%): $max_ram_util, Mem (KB): $max_ram_mem | GPU (%): $max_gpu_util, GPU (MB): $max_gpu_mem, GPU (°C): $max_gpu_temp"
}



monitor_resources() { 
  pid_server=$1
  pid_client=$2
  gpu_mem_init=$3

  # Get CPU, memory, and GPU usage during execution
  while ps -p $pid_client > /dev/null; do
    #CPU
    cpu_usage=$(ps -p $pid_server -o %cpu | tail -n 1)
    #MEM
    mem_usage=$(ps -p $pid_server -o %mem,rss | tail -n 1) #%cpu rss (KB)
    #GPU
    gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits)
    gpu_util=$(echo $gpu_usage  | cut -d ',' -f1)
    gpu_mem_acum=$(echo $gpu_usage  | cut -d ',' -f2)
    gpu_mem_used=$(echo "$gpu_mem_acum - $gpu_mem_init" | bc)
    gpu_temp=$(echo "$gpu_usage"  | cut -d ',' -f3)

    # Save resource usage in a temporary file for evaluation
    echo -e $cpu_usage$DELIMITER_USAGE$mem_usage$DELIMITER_USAGE"$gpu_util\t$gpu_mem_used\t$gpu_temp" >> $USAGE_TEMP_FILE
    sleep 1
  done
  
  all_usage=$(cat $USAGE_TEMP_FILE)                    # Get all usage from temp
  metrics_usage=$(calculate_metrics "$all_usage") # Calculate avg and maximus
  rm $USAGE_TEMP_FILE
	
  echo "$all_usage$DELIMITER$metrics_usage"
}

# Run a model with a prompt and measure
exec_model() {
  model="$1"
  prompt="$2"
 
  # Before start execution, get gpu memory used
  gpu_mem_init=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

  # Run the model and capture both the output of the model and the time (2nd plane)
  { time python3 "../$CLIENT_SCRIPT" "$prompt"; } > $RESULT_TEMP_FILE 2>&1 &
  
  #pid_ollama=$(ps | grep "ollama" | awk '{print $1}')
  pid_time=$!
  pid_server=$(pgrep -f $SERVER_SCRIPT)
  pid_client=$(pgrep -f $CLIENT_SCRIPT)
  total_usage=""
  
  if [ ! -z "$pid_client" ]; then # Check if pid ollama exist (maybe the process has finished already)
    # Collect resource usage while the model is running
    total_usage=$(monitor_resources "$pid_server" "$pid_client" "$gpu_mem_init") 
  fi 
  
  wait $pid_time

  # Kill the gpu process
  pidgpu_ollama=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  sudo kill -9 $pidgpu_ollama # Free gpu resources
  sleep 1 # Wait for normalize resources

  # Separate model output and runtime (the execution time will always be the last 3 lines)
  result=$(cat $RESULT_TEMP_FILE) 
  response=$(echo "$result" | head -n -3) # Everything but the last 3 lines
  time=$(echo "$result" | tail -n 3)      # Last 3 lines only
  rm $RESULT_TEMP_FILE
  
  echo "$response$DELIMITER$time$DELIMITER$total_usage"
}

# For each model, run it with a prompt
for model in "${MODELS[@]}"; do
  for temp in "${TEMPS[@]}"; do
    echo -e "\nModel: $model with t = $temp °"
    python3 ../$SERVER_SCRIPT $model $temp > $SERVER_LOG_FILE 2>&1 &
    sleep 5
    RAG_PID=$(pgrep -f $SERVER_SCRIPT)
    for prompt in "${PROMPTS[@]}"; do 
      echo "Prompt: $prompt"
      echo "---------------------------------"
      
      output=$(exec_model "$model" "$prompt")

      # Separate the results using the global delimiter and redirect them to separate files
      response=""#$(echo "$output" | cut -d"$DELIMITER" -f1)      # Get answer
      time=""#$(echo "$output" | cut -d"$DELIMITER" -f2)          # Get runtime
      all_usage=""#$(echo "$output" | cut -d"$DELIMITER" -f3)     # Get tesource usage
      metrics_usage=""#$(echo "$output" | cut -d"$DELIMITER" -f4) # Get usage metrics 
    
      IFS='~' read -r -d '' response time all_usage metrics_usage <<< "$output"
      
      # Store response
      echo -e "\n-Prompt: $prompt\n" >> $RESPONSE_FOLDER/responses-$model-$temp.txt
      echo "$response" >> $RESPONSE_FOLDER/responses-$model-$temp.txt
      
      # Store continuous usage
      echo -e "\n-Prompt: $prompt\n" >> $MEASURE_FOLDER/all_usage-$model-$temp.txt
      echo "$all_usage" >> $MEASURE_FOLDER/all_usage-$model-$temp.txt

      # Store metrics usage
      echo -e "\n-Prompt: $prompt\n" >> $MEASURE_FOLDER/metrics_usage-$model-$temp.txt
      echo "$time" >> $MEASURE_FOLDER/metrics_usage-$model-$temp.txt
      echo "$metrics_usage" >> $MEASURE_FOLDER/metrics_usage-$model-$temp.txt
    done
    kill -9 $RAG_PID
  done
done

