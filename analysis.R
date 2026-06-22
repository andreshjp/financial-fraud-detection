# ============================================================
# financial-fraud-detection
# Author: Andrés Jiménez | github.com/andreshjp
# ============================================================

# --------------------------
# Carga y preparación de datos
# --------------------------
# Cargar librerías necesarias
library(tidyverse)
library(caret)
library(glmnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(pROC)
library(smotefamily)
library(ggplot2)
library(kernlab)
library(viridis)
library(doParallel)
library(beepr)
library(ggplot2)
library(scales)
library(dplyr)

# Cargar los datos
financial_data <- read.csv("financial_data.csv")

# Exploración inicial
str(financial_data)
summary(financial_data)
lapply(financial_data,unique)
colSums(is.na(financial_data))

# Variables a eliminar con justificación:
financial_data <- financial_data %>%
  select(-c(
    transaction_id,  # ID único que no aporta información
    ip_address,      # Datos demasiado granular, podría crear problemas de privacidad/uso
    device_hash,     # Similar a IP, demasiado específico sin procesamiento adicional
    fraud_type,      # Solo tiene información cuando is_fraud=1 y es la misma siempre
    location,        # Ya hay otra variable que captura esto (geo_anomaly_score)
    device_used,     # Ya hay otra variable que captura esto (payment_channel)
  ))

# Transformación de variables
# Convertir variables categóricas a factores
financial_data <- financial_data %>%
  mutate(
    is_fraud = as.factor(ifelse(is_fraud == "True", 1, 0)),
    transaction_type = as.factor(transaction_type),
    merchant_category = as.factor(merchant_category),
    payment_channel = as.factor(payment_channel)
  )

# Convertir timestamp a formato datetime y extraer características útiles
financial_data <- financial_data %>%
  mutate(
    timestamp = as.POSIXct(timestamp, format = "%Y-%m-%dT%H:%M:%OS"),
    hour_of_day = hour(timestamp),
  ) %>%
  select(-timestamp)

# Imputación de datos faltantes
financial_data <- financial_data %>%
  mutate(
    # Condición 1: Imputación por usuario (si es posible)
    time_since_last_transaction = ifelse(
      is.na(time_since_last_transaction),
      ave(time_since_last_transaction, sender_account, 
          FUN = function(x) median(x, na.rm = TRUE)),
      time_since_last_transaction
    ),
    
    # Condición 2: Imputación global para los restantes
    time_since_last_transaction = ifelse(
      is.na(time_since_last_transaction),
      median(time_since_last_transaction, na.rm = TRUE),
      time_since_last_transaction
    )
  )

# Luego de imputar los datos, eliminar las cuentas porque no tienen valor predictivo real:
financial_data <- financial_data %>% 
  select(-sender_account, -receiver_account)

# Discretización del tiempo
financial_data <- financial_data %>%
  mutate(
    time_since_last_transaction = case_when(
      time_since_last_transaction < 60 ~ "<1 min",
      time_since_last_transaction < 300 ~ "1-5 min",
      time_since_last_transaction < 600 ~ "5-10 min",
      time_since_last_transaction < 1800 ~ "10-30 min",
      time_since_last_transaction < 3600 ~ "30-60 min",
      TRUE ~ "60+ min"
    ) %>% factor(levels = c("<1 min", "1-5 min", "5-10 min", 
                            "10-30 min", "30-60 min", "60+ min"))
  )

# Primero dividir en train/test (70/30)
set.seed(123)
train_index <- createDataPartition(financial_data$is_fraud, p = 0.7, list = FALSE)
train_data <- financial_data[train_index, ]
test_data <- financial_data[-train_index, ]

# Luego balancear SOLO el conjunto de entrenamiento
prop.table(table(train_data$is_fraud))
set.seed(123)
non_fraud_train <- train_data %>% 
  filter(is_fraud == 0) %>% 
  sample_n(sum(train_data$is_fraud == 1))

balanced_train <- bind_rows(
  non_fraud_train,
  train_data %>% filter(is_fraud == 1)
)

# -----------------------
# Modelado Predictivo
# -----------------------

# Configuración común para todos los modelos
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Métricas de evaluación
eval_metrics <- function(model, test_data) {
  pred <- predict(model, newdata = test_data)
  probs <- predict(model, newdata = test_data, type = "prob")
  
  cm <- confusionMatrix(pred, test_data$is_fraud, positive = "1")
  roc <- roc(test_data$is_fraud, probs$"1")
  
  list(
    confusion_matrix = cm,
    roc = roc,
    auc = auc(roc)
  )
}

# Asegurar que los niveles del factor sean correctos (0 = No, 1 = Sí)
balanced_train$is_fraud <- factor(balanced_train$is_fraud,
                                  levels = c("0", "1"),
                                  labels = make.names(c("No", "Yes")))

test_data$is_fraud <- factor(test_data$is_fraud,
                             levels = c("0", "1"),
                             labels = make.names(c("No", "Yes")))

# 1. Regresión Logística
# Regresión Logística simple
set.seed(123)
logit_model <- train(is_fraud ~ .,
                     data = balanced_train,
                     method = "glm",
                     family = "binomial",
                     trControl = ctrl,
                     metric = "ROC")


# Regresión Logística con regularización (Elastic Net)
set.seed(123)
logit_elastic <- train(is_fraud ~ .,
                       data = balanced_train,
                       method = "glmnet",
                       family = "binomial",
                       tuneGrid = expand.grid(alpha = seq(0, 1, 0.1),
                                              lambda = 10^seq(-3, 0, length = 20)),
                       trControl = ctrl,
                       metric = "ROC")



# Evaluar modelos de regresión logística
logit_pred <- predict(logit_model, newdata = test_data)
logit_elastic_pred <- predict(logit_elastic, newdata = test_data)

confusionMatrix(logit_pred, test_data$is_fraud, positive = "Yes")
confusionMatrix(logit_elastic_pred, test_data$is_fraud, positive = "Yes")

# Curvas ROC
logit_probs <- predict(logit_model, newdata = test_data, type = "prob")
logit_elastic_probs <- predict(logit_elastic, newdata = test_data, type = "prob")

roc_logit <- roc(test_data$is_fraud, logit_probs$Yes)
roc_logit_elastic <- roc(test_data$is_fraud, logit_elastic_probs$Yes)



# 2. Árbol de Decisión
# Árbol de decisión con ajuste de hiperparámetros
set.seed(123)
tree_model <- train(is_fraud ~ .,
                    data = balanced_train,
                    method = "rpart",
                    tuneLength = 10,
                    trControl = ctrl,
                    metric = "ROC")

# Visualizar el árbol
prp(tree_model$finalModel, extra = 1, varlen = 0, faclen = 0)

# Evaluación
tree_pred <- predict(tree_model, newdata = test_data)
tree_probs <- predict(tree_model, newdata = test_data, type = "prob")
confusionMatrix(tree_pred, test_data$is_fraud, positive = "Yes")

roc_tree <- roc(test_data$is_fraud, tree_probs$Yes)



# 3. Random Forest
# Configurar paralelización (4 núcleos para equilibrio entre velocidad y RAM)
cl <- makeCluster(4)  
registerDoParallel(cl)

# Random Forest con ajuste de hiperparámetros
set.seed(123)
rf_model <- train(is_fraud ~ .,
                  data = balanced_train,
                  method = "rf",
                  tuneGrid = data.frame(mtry = c(3, 5, 7)),
                  ntree = 200,
                  importance = TRUE,
                  trControl = ctrl,
                  metric = "ROC")

# Importancia de variables
varImpPlot(rf_model$finalModel)

# Evaluación
rf_pred <- predict(rf_model, newdata = test_data)
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")
confusionMatrix(rf_pred, test_data$is_fraud, positive = "Yes")

roc_rf <- roc(test_data$is_fraud, rf_probs$Yes)
# Al final del script, cierra el cluster:
stopCluster(cl)



# 4. Support Vector Machine (SVM)
# SVM con ajuste de hiperparámetros
set.seed(123)
svm_model <- train(is_fraud ~ .,
                   data = balanced_train[sample(nrow(balanced_train), 10000), ],
                   method = "svmLinear",
                   tuneGrid = expand.grid(C = c(0.01, 0.1, 1, 10)),
                   trControl = ctrl,
                   metric = "ROC")

# Evaluación
svm_pred <- predict(svm_model, newdata = test_data)
svm_probs <- predict(svm_model, newdata = test_data, type = "prob")
confusionMatrix(svm_pred, test_data$is_fraud, positive = "Yes")

roc_svm <- roc(test_data$is_fraud, svm_probs$Yes)




# 5. Métodos de Ensamble (Stacking)
# Crear un ensamble de modelos
set.seed(123)
ensemble_models <- list(
  logistic = logit_model,
  elastic = logit_elastic,
  tree = tree_model,
  rf = rf_model,
  svm = svm_model
)

# Función para obtener predicciones de los modelos
get_stacking_data <- function(models, data) {
  preds <- lapply(models, function(model) {
    predict(model, newdata = data, type = "prob")[, "Yes"]
  })
  stack_data <- as.data.frame(preds)
  stack_data$is_fraud <- data$is_fraud
  return(stack_data)
}

# Crear datos para stacking
train_stack <- get_stacking_data(ensemble_models, balanced_train)
test_stack <- get_stacking_data(ensemble_models, test_data)

# Entrenar modelo meta (logistic regression)
stack_model <- train(
  is_fraud ~ .,
  data = train_stack,
  method = "glmnet",
  family = "binomial",
  tuneGrid = expand.grid(alpha = 0.5, lambda = 0.01),
  trControl = ctrl,
  metric = "ROC"
)

# Evaluación del ensamble
stack_pred <- predict(stack_model, newdata = test_stack)
stack_probs <- predict(stack_model, newdata = test_stack, type = "prob")
confusionMatrix(stack_pred, test_stack$is_fraud, positive = "Yes")

roc_stack <- roc(test_stack$is_fraud, stack_probs$"Yes")



# -----------------------
# Comparacion de Modelos
# -----------------------

# Graficar la comparacion de los modelos
plot(roc_logit, col = viridis(6)[1], 
     main = "Comparación de Modelos - Curvas ROC",
     xlab = "Tasa de Falsos Positivos (1 - Especificidad)",
     ylab = "Tasa de Verdaderos Positivos (Sensibilidad)")

# Añadir las demás curvas ROC
plot(roc_logit_elastic, col = viridis(6)[2], add = TRUE)
plot(roc_tree, col = viridis(6)[3], add = TRUE)
plot(roc_rf, col = viridis(6)[4], add = TRUE)
plot(roc_svm, col = viridis(6)[5], add = TRUE)
plot(roc_stack, col = viridis(6)[6], add = TRUE)

# Leyenda horizontal en la parte inferior
legend("bottom", 
       legend = c("Logistic", "Elastic Net", "Decision Tree",
                  "Random Forest", "SVM", "Ensemble"),
       col = viridis(6),
       lwd = 3,
       cex = 0.8,
       bty = "n",
       horiz = TRUE,
       inset = c(0, -0.2),
       xpd = TRUE)

# COMPARACIÓN DE MODELOS
# Crear resumen de métricas
results <- resamples(list(
  Logistic = logit_model,
  ElasticNet = logit_elastic,
  DecisionTree = tree_model,
  RandomForest = rf_model,
  SVM = svm_model,
  Ensemble = stack_model
))

# Resumen estadístico
summary(results)

# Gráfico de comparacaión
bwplot(results, metric = "ROC")

# Calcular los valores AUC para cada modelo
auc_values <- c(
  auc(roc_logit),           
  auc(roc_logit_elastic),   
  auc(roc_tree),            
  auc(roc_rf),              
  auc(roc_svm),            
  auc(roc_stack)            
)
# Identificar el mejor modelo
best_model <- ifelse(which.max(auc_values) == 1, "Logistic Regression",
                     ifelse(which.max(auc_values) == 2, "Elastic Net",
                            ifelse(which.max(auc_values) == 3, "Decision Tree",
                                   ifelse(which.max(auc_values) == 4, "Random Forest",
                                          ifelse(which.max(auc_values) == 5, "SVM", "Ensemble")))))

cat("El mejor modelo es:", best_model, "con un AUC de", max(auc_values), "\n")

# Análisis de Variables Importantes
elastic_imp <- varImp(logit_elastic, scale = FALSE)$importance
elastic_imp_df <- data.frame(
  Variable = rownames(elastic_imp),
  Importance = elastic_imp$Overall
) %>%
  arrange(desc(Importance)) %>%
  head(5)
print(elastic_imp_df)

# Graficar importancia
ggplot(elastic_imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 5 Variables más importantes - Elastic Net",
       x = "Variable",
       y = "Importancia") +
  theme_minimal()

# Otros Gráficos

# Preparar datos
fraude_por_tiempo <- balanced_train %>%
  group_by(time_since_last_transaction, is_fraud) %>%
  summarise(n = n(), .groups = 'drop') %>%
  mutate(prop = n / sum(n)) %>%
  filter(is_fraud == "Yes")

# Gráfico
ggplot(fraude_por_tiempo, aes(x = time_since_last_transaction, y = prop, fill = time_since_last_transaction)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = scales::percent(prop)), vjust = -0.5) +
  scale_fill_manual(values= c("#A6CEE3", "#1F78B4", "#BFD3E6", "#8C96C6", "#8856A7", "#225EA8")) +
  labs(title = "Distribución de Transacciones Fraudulentas por Intervalo de Tiempo",
       x = "Tiempo desde última transacción",
       y = "Porcentaje de casos fraudulentos",
       fill = "Intervalo") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Crear datos agregados
fraud_heatmap_data <- balanced_train %>%
  mutate(hour_group = cut(hour_of_day, 
                          breaks = c(0, 4, 8, 12, 16, 20, 24),
                          labels = c("00:00-04:00", "04:00-08:00", 
                                     "08:00-12:00", "12:00-16:00", 
                                     "16:00-20:00", "20:00-24:00"))) %>%
  group_by(hour_group, payment_channel, is_fraud) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(hour_group, payment_channel) %>%
  mutate(prop = n/sum(n)) %>%
  filter(is_fraud == "Yes")

# Gráfico de mapa de calor
ggplot(fraud_heatmap_data, aes(x = hour_group, y = payment_channel, fill = prop)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "#ecf0f1", high = "#3498db", 
                      labels = percent_format()) +
  labs(title = "Tasa de fraude por franja horaria y canal de pago",
       subtitle = "Picos de fraude en horario nocturno y canales alternativos",
       x = "Franja horaria",
       y = "Canal de pago",
       fill = "% de Fraude") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(color = "gray40"))