r_obj <- readRDS("data/UROMOL_TaLG.teachingcohort.rds")

# If the object is a data frame, view the first few rows
head(r_obj)

# To check the structure of the object
str(r_obj)

expr <- r_obj$exprs
expr <- data.frame(expr)
write.csv(expr, file="./data/expr.csv")