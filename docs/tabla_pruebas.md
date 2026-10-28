# Tabla de Pruebas Obligatoria

Según las especificaciones del proyecto, se debe registrar una tabla de pruebas que incluya:
1. Una consulta parcial (evidencia incompleta / variables no proporcionadas o desconocidas).
2. Una consulta completa (todos los atributos válidos proporcionados).
3. Una entrada inválida (manejo y rechazo controlado de errores).

---

| ID Prueba | Tipo de Caso | Entrada / Evidencia Proporcionada | Resultado Esperado | Resultado Obtenido | Estado | Observaciones / Explicación |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-01** | **Consulta Parcial** | • `VisitorType = "Returning_Visitor"`<br>• `ProductRelated = Desconocido`<br>• `ProductRelated_Duration = Desconocido` | Estimación basada exclusivamente en el tipo de visitante mediante marginalización en Naive Bayes. Probabilidad esperada: ~$0.1398$. | *Pendiente de ejecución* | *Planificado* | Las variables no proporcionadas se tratan como desconocidas (factor neutro $1.0$). |
| **TC-02** | **Consulta Completa** | • `VisitorType = "Returning_Visitor"`<br>• `ProductRelated = 35` (alto)<br>• `ProductRelated_Duration = 1500.0` (alto) | Estimación en 3 etapas progresivas (Naive Bayes) y evaluación directa en RF y MLP. Probabilidad esperada $> 0.30$. | *Pendiente de ejecución* | *Planificado* | Demostración de las dos versiones: discretizada para Bayes y continua para ML. |
| **TC-03** | **Entrada Inválida** | • `VisitorType = "Returning_Visitor"`<br>• `ProductRelated = -5`<br>• `ProductRelated_Duration = "duracion_invalida"` | Rechazo explícito de la entrada con mensaje descriptivo de error en GUI/CLI. No debe colapsar la aplicación. | *Pendiente de ejecución* | *Planificado* | Validación estricta de rangos no negativos y tipos numéricos. |
