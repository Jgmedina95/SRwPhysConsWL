import SymbolicRegression: SRRegressor, Options, EquationSearch, calculate_pareto_frontier
using SymbolicRegression
using DataFrames
using CSV
using LinearAlgebra
using Statistics
include("PhConstraints.jl")

df2 = CSV.read("Aerobic_AugmentedLactate0_100_Temp30_v3.csv", DataFrame, header = true, delim = ',')
Aerobic_X = df2[df2.Aerobic .== 1, ["Time","Concentration"]]
#rename Concentration to Conc
rename!(Aerobic_2_X, :Concentration => :Conc)
Aerobic_y = df2[df2.Aerobic .== 1, ["Value"]]
#remove concentration at 80,90,100 Because noise is too high. 
#And 0,10 and 20 because they are repeated in the low concentration dataset.

Aerobic_X,Aerobic_y = Aerobic_X[Aerobic_X.Conc .!= 20, :], Aerobic_y[Aerobic_X.Conc .!= 20, :]
Aerobic_X,Aerobic_y = Aerobic_X[Aerobic_X.Conc .!= 0, :],  Aerobic_y[Aerobic_X.Conc .!= 0, :]
Aerobic_X,Aerobic_y = Aerobic_X[Aerobic_X.Conc .!= 10, :], Aerobic_y[Aerobic_X.Conc .!= 10, :]
Aerobic_X,Aerobic_y = Aerobic_X[Aerobic_X.Conc .!= 50, :], Aerobic_y[Aerobic_X.Conc .!= 50, :]
Aerobic_X,Aerobic_y = Aerobic_X[Aerobic_X.Conc .!= 40, :], Aerobic_y[Aerobic_X.Conc .!= 40, :]
Aerobic_X,Aerobic_y = Aerobic_X[Aerobic_X.Conc .!= 30, :], Aerobic_y[Aerobic_X.Conc .!= 30, :]
df1 = CSV.read("Aerobic_AugmentedLowConc_02.csv", DataFrame, header = true, delim = ',')

Aerobic_X = df1[df1.Aerobic   .== 0, ["Time","Conc"]]
Aerobic_y = df1[df1.Aerobic   .== 0, ["Value"]]

#Removing 17.5 concentration because data is too noisy.
Aerobic_X,Aerobic_y = Aerobic_X[Aerobic_X.Conc .!= 17.5, :], Aerobic_y[Aerobic_X.Conc .!= 17.5, :]

#concatenate the two datasets
Aerobic_X = vcat(Aerobic_low_X, Aerobic_X)
Aerobic_y = vcat(Aerobic_low_y, Aerobic_y)

pow2(x) = x^2
pow3(x) = x^3
sigmoid(x) = 1/(1+exp(-x))
nested_exp(x) = exp(-exp(x))

loss_fuction = ConstrainsData.Asymptote_loss   #Physical Constraint 

#In the following, 
options_1 = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[exp,log,sqrt,pow2, pow3,nested_exp],
    custom_loss_function = loss_fuction,
    enable_autodiff=true,
    npopulations=100,
    return_state=true
)

options_2 = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[exp,log,sqrt,pow2, pow3,nested_exp],
    npopulations=100
)

#data_names = [ "Aerobic","Anaerobic" ]
data_names = ["Low_Conc"]
#datasets = zip([ Aerobic_X,Anaerobic_X],[Aerobic_y, Anaerobic_y], )
datasets = [(Aerobic_X, Aerobic_y)]
for ((X, y),name) in zip(datasets, data_names)
    X = Transpose(Matrix(X))
    y = Matrix(y)
    y = Vector(y[:,1])
    for i in 1:3
        println("Running dataset $name, iteration $i")
        hof = EquationSearch(
            X, y, niterations=10, options=options_1,
            parallelism=:multithreading
        )
        dataset = SymbolicRegression.Dataset(X,y)
        SymbolicRegression.LossFunctionsModule.update_baseline_loss!(dataset,options_2)
        for population in hof[1][1]
            for mem in population.members
            mem.score, mem.loss = SymbolicRegression.PopMemberModule.score_func(dataset,mem.tree,options_2)
            end
            end
        hof2 = EquationSearch(X,y;niterations=90,options=options_2,saved_state=hof);
        #global dominating = calculate_pareto_frontier(train_X,train_y,hof2,QBC.options2);
        #change the name of hall_of_fame.csv to something else
        #change the name of the file that starts with hall_of_fame and ends with csv
        #find current name of hall_of_fame.csv
        files = readdir()
        for file in files
            if occursin("hall_of_fame.csv", file) && occursin("csv", file) && !occursin(".bkup", file)
                println(file)
                mv(file, "hall_of_fame_$(i)_$(name)_Aer_fulldataset_w_low_conc.csv")
            end
        end
        #mv("hall_of_fame.csv", "hall_of_fame_$(i)_$(name).csv")
    end
end