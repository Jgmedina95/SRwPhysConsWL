#functions
module ConstrainsData
using SymbolicRegression 
using Random
using Distributions

export Asymptote_loss

function symmetry_loss(tree::SymbolicRegression.Node, dataset::SymbolicRegression.Dataset{T},options,var1=1,var2=2,var3=3,var4=4,n=100) where {T}
   _,d= size(dataset.X)
   symmetrydata = copy(dataset.X)
   symmetrydata[var1,:],symmetrydata[var2,:],symmetrydata[var3,:],symmetrydata[var4,:]=symmetrydata[var2,:],symmetrydata[var1,:],symmetrydata[var4,:],symmetrydata[var3,:]
   prediction1, complete1 = SymbolicRegression.eval_tree_array(tree,dataset.X,options)
   (!complete1) && return(T(10000000))
   prediction2, complete2 = SymbolicRegression.eval_tree_array(tree,symmetrydata,options)
   (!complete2) && return(T(10000000))
   
   predictive_loss_L2Dis = sum(abs.(dataset.y .- prediction1))
   symmetry_loss = sum(n*abs.(prediction1-prediction2))/d
  
   return predictive_loss_L2Dis + symmetry_loss

end

function divergency_symmetry_loss(treetree::SymbolicRegression.Node, dataset::SymbolicRegression.Dataset{T},options,var1=2,var2=3,n=5) where {T}
   for i in collect(1:d)
      divergency_data[dir,i] = 7
      divergency_data[dir+1,i] = 7
   end
   _,d= size(dataset.X)
   symmetrydata = copy(dataset.X)
   symmetrydata[var1,:],symmetrydata[var2,:]=symmetrydata[var2,:],symmetrydata[var1,:]
   prediction1, complete1 = SymbolicRegression.eval_tree_array(tree,dataset.X,options)
   (!complete1) && return(T(10000000))
   prediction2, complete2 = SymbolicRegression.eval_tree_array(tree,symmetrydata,options)
   (!complete2) && return(T(10000000))
   
   prediction_div, _ = SymbolicRegression.eval_tree_array(tree, divergency_data, options)

   predictive_loss_L2Dis = sum(abs.(dataset.y .- prediction1))
   symmetry_loss = n*sum(abs.(prediction1-prediction2))/d
   divergency_loss = n*sum(isfinite.(prediction_div))/d      #if Inf then no addition to divergency_loss

   return predictive_loss_L2Dis+ symmetry_loss+ divergency_loss
end 
   function Asymptote_loss(tree::SymbolicRegression.Node , dataset::SymbolicRegression.Dataset{T}, options;dir=1,n=500) where {T}
      _,d = size(dataset.X)
      ## Extract time and concentration columns
      num_columns_to_keep = 50 # Adjust this based on how many columns you want
      linear_interp_values = LinRange(3500, 6000, num_columns_to_keep)
      
      # Get the total number of columns in dataset.X
      
      # Randomly sample 'num_columns_to_keep' unique column indices
      random_columns = randperm(d)[1:num_columns_to_keep]
      
      # Create a downsampled version of dataset.X with the selected columns
      downsampled_X = dataset.X[:, random_columns]
      downsampled_X[dir,:] = linear_interp_values
      tree_asymptote = fill(Float32(0.5),num_columns_to_keep)
      
      if n!=0
         prediction, complete = SymbolicRegression.eval_tree_array(tree, dataset.X, options)
         (!complete) && return T(10000000)
      end
      if n!=0
            _,der_asymp, complete_asymp = try SymbolicRegression.eval_diff_tree_array(tree,downsampled_X,options,dir) catch; return T(10000000) end
            (!complete_asymp) && return T(10000000)
      end
         predictive_loss_L2Dis = sum(abs.(dataset.y .- prediction).^2)
         asymptotic_loss = n*sum(max.(0,(abs.(der_asymp) .- tree_asymptote)))
         return predictive_loss_L2Dis + asymptotic_loss
   end
    
    function divergency(tree::SymbolicRegression.Node , dataset::SymbolicRegression.Dataset{T}, options;dir=1,n=100) where {T}
      _,d = size(dataset.X)
      divergency_data = copy(dataset.X)
      for i in collect(1:d)
         divergency_data[dir,i] = 7
	 divergency_data[dir+1,i] = 7
         divergency_data[dir+2,i] = 19
         divergency_data[dir+3,i] = 19   
   end
       prediction, complete = SymbolicRegression.eval_tree_array(tree, dataset.X, options)
       (!complete) && return T(10000000)
 prediction_div, _ = SymbolicRegression.eval_tree_array(tree, divergency_data, options)
     
      predictive_loss_L2Dis = sum(abs.(dataset.y .- prediction).^2)
      divergency_loss = n*sum(isfinite.(prediction_div))/d      #if Inf then no addition to divergency_loss
      return predictive_loss_L2Dis + divergency_loss
      end
            

    function Monotone_loss(tree::SymbolicRegression.Node , dataset::SymbolicRegression.Dataset{T}, options;dir=3,n=100) where {T}
    
    prediction, derivative, complete = SymbolicRegression.eval_diff_tree_array(tree, dataset.X, options, dir)
    (!complete) && return T(10000000)
    predictive_loss_L2Dis = sum((abs.(dataset.y .- prediction)).^2)
    ph_loss = sum(n*max.(derivative,0))
    return predictive_loss_L2Dis + ph_loss
    end

end
