using Agents
using Random
using CSV
using Distributions
using DataFrames
using Plots
using GLMakie

space = GridSpace((20, 20); periodic = false)

#Helper functions

# Right-skewed distribution
function random_human_age()
    age = Int(trunc(exp(rand(Normal(log(30), 1.5)))))
    return age > 120 ? 120 : age
end

function random_australian_age()
    # Assign probabilities for each age group based on ABS data
    # https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/sep-2021
    v = [0.025, 0.028, 0.028, 0.031, 0.032, 0.033, 0.032, 0.033, 0.035, 0.037, 0.039, 0.042, 0.043, 0.045, 0.047, 0.049, 0.051, 0.054, 0.054, 0.053, 0.046, 0.036, 0.027, 0.015, 0.005]
    age_groups = collect(0:4:120)

    s = sum(v)
    probs = v ./ s

    age_group = rand(Categorical(probs))
    age = rand(age_groups[age_group]:age_groups[age_group]+4)
    return trunc(Int, age)
end

# 1 = Male, -1 = Female
function random_human_gender()
    gender = rand(Bernoulli(0.5)) == 1 ? 1 : 0
    return gender
end

function random_australian_education(age::Int)
    if age < 15
        return 1 # No educational qualification
    elseif age < 20
        v = [0.536, 0.248, 0.086, 0.064, 0.035, 0.031]
    elseif age < 25
        v = [0.232, 0.308, 0.190, 0.090, 0.095, 0.085]
    elseif age < 35
        v = [0.072, 0.104, 0.350, 0.181, 0.139, 0.155]
    elseif age < 45
        v = [0.058, 0.060, 0.305, 0.230, 0.169, 0.178]
    elseif age < 55
        v = [0.042, 0.035, 0.274, 0.254, 0.169, 0.226]
    elseif age < 65
        v = [0.039, 0.029, 0.219, 0.284, 0.172, 0.257]
    else
        v = [0.848, 0.021, 0.042, 0.029, 0.030, 0.030]
    end
    
    s = sum(v)
    probs = v ./ s

    # Generate a random education level based on the probabilities
    education = rand(Categorical(probs))

    # Return the education level as a category number
    # 1: No educational qualification
    # 2: Primary education
    # 3: Secondary education
    # 4: Certificate level
    # 5: Advanced diploma or diploma
    # 6: Bachelor's degree or higher
    return education
end

function random_australian_employment(age)
    # Assign probabilities for each employment status based on ABS data
    # https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia/latest-release
    if age >= 65
        v = [0.070, 0.023, 0.510, 0.064, 0.315, 0.018, 0.000]
    elseif age >= 25
        v = [0.692, 0.051, 0.050, 0.022, 0.100, 0.084, 0.001]
    else
        v = [0.721, 0.048, 0.010, 0.027, 0.097, 0.092, 0.005]
    end
    
    s = sum(v)
    probs = v ./ s

    # Generate a random employment status based on the probabilities
    employment = rand(Categorical(probs))

    # Return the employment status as a category number
    # 1: Employed - full time
    # 2: Employed - part time
    # 3: Unemployed - actively looking for work
    # 4: Unemployed - not actively looking for work
    # 5: Not in the labour force - retired
    # 6: Not in the labour force - studying or training
    # 7: Not in the labour force - other reasons
    return employment
end

function generate_income(age::Int, gender::Int, education::Int, employment::Int)
    # Calculate base income based on age and gender
    if age < 15
        base_income = 10000 # Invalid age
    elseif age < 20
        base_income = 20000 + 5000 * gender # Age 15-19
    elseif age < 25
        base_income = 30000 + 5000 * gender # Age 20-24
    elseif age < 35
        base_income = 50000 + 10000 * gender # Age 25-34
    elseif age < 45
        base_income = 65000 + 10000 * gender # Age 35-44
    elseif age < 55
        base_income = 75000 + 10000 * gender # Age 45-54
    elseif age < 65
        base_income = 75000 + 5000 * gender # Age 55-64
    else
        base_income = 50000 # Age 65-85
    end
    
    # Adjust income based on education level
    if education < 4
        base_income *= 0.7 # No post-school qualification
    elseif education <6
        base_income *= 1.0 # Post-school qualification below bachelor level
    elseif education == 6
        base_income *= 1.3 # Bachelor degree or above
    else
        base_income = 30000 # Invalid education level
    end
    
    # Adjust income based on employment status
    if employment == 1
        base_income *= 1.2 # Full-time
    elseif employment == 2
        base_income *= 0.8 # Part-time
    elseif employment == 3 || employment == 4
        base_income *= 0.5 # Unemployed
    elseif employment > 4
        base_income *= 0.6 # Not in the labour force
    else
        base_income = 30000 # Invalid employment status
    end
    
    # Add random noise to income
    noise = rand(Normal(0, 0.2 * base_income)) # 20% standard deviation
    income = round(base_income + noise)
    income = trunc(Int, income)
    
    # Return income as integer
    return max(income, 0) # Ensure income is non-negative
end

# Caloric Intake and Incidental Exercise
function random_caloric_intake(gender::Int)
    if gender == 1
        intake = clamp(rand(Normal(8800,1500)), 2000, 15000)
    else
        intake = clamp(rand(Normal(6500,1200)), 2000, 12000)
    end
    return intake
end

function random_incidental_exercise()
    μ = 837  # mean daily incidental exercise in kJ
    σ = 150  # standard deviation of daily incidental exercise in kJ
    return clamp(rand(Normal(μ, σ)), 0, 2000)
end

# Generate random positions in the grid
function generate_random_position(grid::GridSpace)
    x = rand(1:size(grid)[1])
    y = rand(1:size(grid)[2])
    return (x, y)
end

@agent TransportAgent GridAgent{2} begin
    #DEMOGRAPHICS 
    age::Int
    gender::Int
    education::Int
    employment::Int
    income::Int

    #TRANSPORT LAYER
    original_transport_type ::Int64 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking BUT ALL START WITH 0. 
    transport_type::Int64 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking 
    transport_choice::Int64 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking, 6 = Ride-hail app BUT ALL START WITH 0. 

    #COGNITIVE LAYER - AUTONOMOUS VEHICLES
    #Theory of planned behaviour
    av_attitudes::Float32
    av_social_norms::Float32
    av_control_factors::Float32
    av_behavioural_intention::Int64
    av_subjective_norm::Float32
    av_facilitating_conditions::Float32
    av_threshold::Float32

    #COGNITIVE LAYER - AUTONOMOUS RIDE HAIL APPS
    #Theory of planned behaviour
    rh_attitudes::Float32
    rh_social_norms::Float32
    rh_control_factors::Float32
    rh_behavioural_intention::Int64
    rh_subjective_norm::Float32
    rh_facilitating_conditions::Float32
    rh_threshold::Float32

    #Additional cognitive factors 
    impulsivity::Float32

    # Cognitive behavioural control - AV
    av_cb_pos::Float32
    av_cb_neg::Float32

    # Cognitive behavioural control - RH
    rh_cb_pos::Float32
    rh_cb_neg::Float32

    # PHYSICAL HEALTH LAYER
    physical_health_layer::Float32
    sedentary_behaviour::Float32
end

function initialize(; total_agents = 250, griddims = (20, 20), private_AV_cost = 20000, rh_trip_cost = 10, seed = 100, av_threshold_model = 5.0, rh_threshold_model = 5.0, AVs = 0, RH_trips = 0, AVs_time_series = [0], # Starting with 0 AVs
    RH_trips_time_series = [0])
    rng = MersenneTwister(seed)
    space = GridSpace(griddims, periodic = false)
    properties = Dict(:private_AV_cost => private_AV_cost, :rh_trip_cost => rh_trip_cost, :tick => 1, :av_threshold_model => av_threshold_model, :rh_threshold_model => rh_threshold_model, :AVs => 0, :RH_trips => 0, :AVs_time_series => [0], :RH_trips_time_series => [0], :total_agents => total_agents)
    model = AgentBasedModel(TransportAgent, space; properties = properties, rng, scheduler = Schedulers.Randomly()
    )

    # Adding the Agents

    # Adding car-driving agents
    for n in 1:total_agents*0.6

        #DEMOGRAPHICS
        age = random_human_age()
        gender = random_human_gender()
        education = random_australian_education(age)
        employment = random_australian_employment(age)
        income = generate_income(age, gender, education, employment)

        #TRANSPORT LAYER
        original_transport_type = 2 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking BUT ALL START WITH 0.
        transport_type = 2 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking BUT ALL START WITH 0. 
        transport_choice = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking, 6 = Ride-hail app, BUT ALL START WITH 0. 

        #COGNITIVE LAYER - AUTONOMOUS VEHICLES
        #Theory of planned behaviour
        av_attitudes = rand(model.rng)
        av_social_norms = rand(model.rng)
        av_control_factors = rand(model.rng)
        av_behavioural_intention = 0
        av_subjective_norm = rand(model.rng)
        av_facilitating_conditions = rand(model.rng)
        av_threshold = rand(model.rng)

        #COGNITIVE LAYER - AUTONOMOUS RIDE-HAIL APPS
        #Theory of planned behaviour
        rh_attitudes = rand(model.rng)
        rh_social_norms = rand(model.rng)
        rh_control_factors = rand(model.rng)
        rh_behavioural_intention = 0
        rh_subjective_norm = rand(model.rng)
        rh_facilitating_conditions = rand(model.rng)
        rh_threshold = rand(model.rng)

        #Additional cognitive factors 
        impulsivity = rand(model.rng)

        # Control behaviours - AV 
        av_cb_pos = rand(model.rng)
        av_cb_neg = rand(model.rng)

        # Control behaviours - RH
        rh_cb_pos = rand(model.rng)
        rh_cb_neg = rand(model.rng)

        # PHYSICAL HEALTH LAYER
        physical_health_layer = rand(model.rng)
        sedentary_behaviour = rand(model.rng)


        add_agent!(TransportAgent, model, age, gender, education, employment, income, original_transport_type, transport_type, transport_choice, av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold, rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg, physical_health_layer, sedentary_behaviour)
    end 

    # Adding PT-using agents 
    for n in 1:total_agents*0.2
        #DEMOGRAPHICS
        age = random_human_age()
        gender = random_human_gender()
        education = random_australian_education(age)
        employment = random_australian_employment(age)
        income = generate_income(age, gender, education, employment)

        #TRANSPORT LAYER
        original_transport_type = 3 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking BUT ALL START WITH 0.
        transport_type = 3 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking BUT ALL START WITH 0. 
        transport_choice = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking, 6 = Ride-hail app, BUT ALL START WITH 0. 

        #COGNITIVE LAYER - AUTONOMOUS VEHICLES
        #Theory of planned behaviour
        av_attitudes = rand(model.rng)
        av_social_norms = rand(model.rng)
        av_control_factors = rand(model.rng)
        av_behavioural_intention = 0
        av_subjective_norm = rand(model.rng)
        av_facilitating_conditions = rand(model.rng)
        av_threshold = rand(model.rng)

        #COGNITIVE LAYER - AUTONOMOUS RIDE-HAIL APPS
        #Theory of planned behaviour
        rh_attitudes = rand(model.rng)
        rh_social_norms = rand(model.rng)
        rh_control_factors = rand(model.rng)
        rh_behavioural_intention = 0
        rh_subjective_norm = rand(model.rng)
        rh_facilitating_conditions = rand(model.rng)
        rh_threshold = rand(model.rng)

        #Additional cognitive factors 
        impulsivity = rand(model.rng)

        # Control behaviours - AV 
        av_cb_pos = rand(model.rng)
        av_cb_neg = rand(model.rng)

        # Control behaviours - RH
        rh_cb_pos = rand(model.rng)
        rh_cb_neg = rand(model.rng)

        # PHYSICAL HEALTH LAYER
        physical_health_layer = rand(model.rng)
        sedentary_behaviour = rand(model.rng)


        add_agent!(TransportAgent, model, age, gender, education, employment, income, original_transport_type, transport_type, transport_choice, av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold, rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg, physical_health_layer, sedentary_behaviour)
    end 

    # Adding cycling agents 
    for n in 1:total_agents*0.1
        #DEMOGRAPHICS
        age = random_human_age()
        gender = random_human_gender()
        education = random_australian_education(age)
        employment = random_australian_employment(age)
        income = generate_income(age, gender, education, employment)

        #TRANSPORT LAYER
        original_transport_type = 4 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking BUT ALL START WITH 0.
        transport_type = 4 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking BUT ALL START WITH 0. 
        transport_choice = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking, 6 = Ride-hail app, BUT ALL START WITH 0. 

        #COGNITIVE LAYER - AUTONOMOUS VEHICLES
        #Theory of planned behaviour
        av_attitudes = rand(model.rng)
        av_social_norms = rand(model.rng)
        av_control_factors = rand(model.rng)
        av_behavioural_intention = 0
        av_subjective_norm = rand(model.rng)
        av_facilitating_conditions = rand(model.rng)
        av_threshold = rand(model.rng)

        #COGNITIVE LAYER - AUTONOMOUS RIDE-HAIL APPS
        #Theory of planned behaviour
        rh_attitudes = rand(model.rng)
        rh_social_norms = rand(model.rng)
        rh_control_factors = rand(model.rng)
        rh_behavioural_intention = 0
        rh_subjective_norm = rand(model.rng)
        rh_facilitating_conditions = rand(model.rng)
        rh_threshold = rand(model.rng)

        #Additional cognitive factors 
        impulsivity = rand(model.rng)

        # Control behaviours - AV 
        av_cb_pos = rand(model.rng)
        av_cb_neg = rand(model.rng)
        
        # Control behaviours - RH
        rh_cb_pos = rand(model.rng)
        rh_cb_neg = rand(model.rng)

        # PHYSICAL HEALTH LAYER
        physical_health_layer = rand(model.rng)
        sedentary_behaviour = rand(model.rng)


        add_agent!(TransportAgent, model, age, gender, education, employment, income, original_transport_type, transport_type, transport_choice, av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold, rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg, physical_health_layer, sedentary_behaviour)
    end 

    # Adding walking for transport agents 
    for n in 1:total_agents*0.1
        #DEMOGRAPHICS
        age = random_human_age()
        gender = random_human_gender()
        education = random_australian_education(age)
        employment = random_australian_employment(age)
        income = generate_income(age, gender, education, employment)

        #TRANSPORT LAYER
        original_transport_type = 5 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking BUT ALL START WITH 0.
        transport_type = 5 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking BUT ALL START WITH 0. 
        transport_choice = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking, 6 = Ride-hail app, BUT ALL START WITH 0. 

        #COGNITIVE LAYER - AUTONOMOUS VEHICLES
        #Theory of planned behaviour
        av_attitudes = rand(model.rng)
        av_social_norms = rand(model.rng)
        av_control_factors = rand(model.rng)
        av_behavioural_intention = 0
        av_subjective_norm = rand(model.rng)
        av_facilitating_conditions = rand(model.rng)
        av_threshold = rand(model.rng)

        #COGNITIVE LAYER - AUTONOMOUS RIDE-HAIL APPS
        #Theory of planned behaviour
        rh_attitudes = rand(model.rng)
        rh_social_norms = rand(model.rng)
        rh_control_factors = rand(model.rng)
        rh_behavioural_intention = 0
        rh_subjective_norm = rand(model.rng)
        rh_facilitating_conditions = rand(model.rng)
        rh_threshold = rand(model.rng)

        #Additional cognitive factors 
        impulsivity = rand(model.rng)

        # Control behaviours - AV 
        av_cb_pos = rand(model.rng)
        av_cb_neg = rand(model.rng)

        # Control behaviours - RH
        rh_cb_pos = rand(model.rng)
        rh_cb_neg = rand(model.rng)

        # PHYSICAL HEALTH LAYER
        physical_health_layer = rand(model.rng)
        sedentary_behaviour = rand(model.rng)


        add_agent!(TransportAgent, model, age, gender, education, employment, income, original_transport_type, transport_type, transport_choice, av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold, rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg, physical_health_layer, sedentary_behaviour)
    end 
    return model
end

model = initialize()

function agent_step!(agent, model)
    av_decision!(agent, model)
    rh_decision!(agent, model)
    agent_health!(agent, model)
end

function model_step!(model)
    model.tick += 1

    if model.tick <= 25
        model.av_threshold_model = 90
        model.rh_threshold_model = 90
    
    elseif model.tick > 20 && model.tick <= 50
        model.av_threshold_model = 75
        model.rh_threshold_model = 80

    elseif model.tick > 50 && model.tick <= 75
        model.av_threshold_model = 50
        model.rh_threshold_model = 50
    
    elseif model.tick > 75 && model.tick <= 90
        model.av_threshold_model = 30
        model.rh_threshold_model = 30

    else 
        model.av_threshold_model = 10
        model.rh_threshold_model = 10
    end

end


# AV DECISION-MAKING # 
function AV_TPB(av_attitudes, av_control_factors, av_social_norms, av_subjective_norm, av_faciliating_conditions, threshold)
    if false
        print("Facil Conditions: ", av_faciliating_conditions, " Sum: ", sum(av_attitudes)+sum(av_control_factors)+sum(av_social_norms)+sum(av_subjective_norm),
            " Threshold: ", threshold, "\n")
    end
    return all(av_faciliating_conditions) &&
        sum(av_attitudes)+
        sum(av_control_factors)+
        sum(av_social_norms)+
        sum(av_subjective_norm) > model.av_threshold_model
    print(return)
end

function av_decision!(agent, model)
    attitudes = agent.av_attitudes
    control_behaviour = agent.av_cb_pos - 2*agent.av_cb_neg
    # calculating social_norms
    subjective_norms = 0
    num_neighbors = 0
    for neighbor in nearby_agents(agent, model)
        subjective_norms = subjective_norms + neighbor.av_attitudes
        num_neighbors = num_neighbors + 1
    end
    
    #Taking the average
    subjective_norms = num_neighbors==0 ? 0 : subjective_norms / num_neighbors

    #Taking the average
    descriptive_norms = model.AVs / model.total_agents

    facil_conditions = [agent.income > model.private_AV_cost]

    AV_decision = AV_TPB(attitudes, control_behaviour, subjective_norms, descriptive_norms, facil_conditions, model.av_threshold_model)
    
    if AV_decision 
        push!(model.AVs_time_series, model.AVs_time_series[end] + 1)
    else
        push!(model.AVs_time_series, model.AVs_time_series[end])
    end

    if AV_decision
        agent.transport_choice = 1
    end

end

# RIDE-HAIL DECISION-MAKING # 

function RH_TPB(rh_attitudes, rh_control_factors, rh_social_norms, rh_subjective_norm, rh_faciliating_conditions, threshold)
    if false
        print("Facil Conditions: ", rh_faciliating_conditions, " Sum: ", sum(rh_attitudes)+sum(rh_control_factors)+sum(rh_social_norms)+sum(rh_subjective_norms),
            " Threshold: ", threshold, "\n")
    end
    return all(rh_faciliating_conditions) &&
        sum(rh_attitudes)+
        sum(rh_control_factors)+
        sum(rh_social_norms)+
        sum(rh_subjective_norm) > threshold
    print(return)
end

function rh_decision!(agent, model)
    attitudes = agent.rh_attitudes
    control_behaviour = agent.rh_cb_pos - 2*agent.rh_cb_neg
    # calculating social_norms
    subjective_norms = 0
    num_neighbors = 0
    for neighbor in nearby_agents(agent, model)
        subjective_norms = subjective_norms + neighbor.rh_attitudes
        num_neighbors = num_neighbors + 1
    end
    
    # Taking the average
    subjective_norms = num_neighbors == 0 ? 0 : subjective_norms / num_neighbors

    # Taking the average
    descriptive_norms = model.RH_trips / model.total_agents

    facil_conditions = [(agent.income*0.001) > model.rh_trip_cost]

    RH_decision = RH_TPB(attitudes, control_behaviour, subjective_norms, descriptive_norms, facil_conditions, model.rh_threshold_model)

    if RH_decision 
        push!(model.RH_trips_time_series, model.RH_trips_time_series[end] + 1)
    else
        push!(model.RH_trips_time_series, model.RH_trips_time_series[end])
    end

    if RH_decision
        agent.transport_choice = 6
    end

end



function agent_health!(agent, model) # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking, 6 = Ride-hail app, BUT ALL START WITH 0. 
    if agent.transport_choice == 1 || agent.transport_choice == 2 || agent.transport_choice == 6
        if agent.original_transport_type == 2
            agent.sedentary_behaviour = agent.sedentary_behaviour
        else 
            agent.sedentary_behaviour += 0.1
            agent.physical_health_layer -= 0.01
        end
    end
end

            



# adata = [:pos, :transport_type, :transport_choice, :age, :income, :original_transport_type, :av_attitudes, :av_social_norms, :av_control_factors, :av_behavioural_intention, :rh_attitudes, :rh_social_norms, :rh_control_factors, :rh_behavioural_intention, :impulsivity, :physical_health_layer, :sedentary_behaviour]
# data, _ = run!(model, agent_step!, model_step!, 365; adata)

# # Simulating the number of AVs and RHs over time - interactive graph doesn't work

# time_ticks = 1:91251

# # Create a figure and axis for the plot
# fig = Figure(resolution = (800, 400))
# ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Number")

# lines!(ax, time_ticks, model.AVs_time_series, linewidth=2, color=:blue, label="AVs")
# lines!(ax, time_ticks, model.RH_trips_time_series, linewidth=2, color=:red, label="RH Trips")

# # Add labels to the axes
# # Makie.xlabel!(ax, "Time")
# # Makie.ylabel!(ax, "Number")
# leg = Legend(fig[1, 2], ax)
# fig[1, 2] = leg

# # Show the plot
# fig





using CairoMakie # using a different plotting backend that enables interactive plots
using Statistics


model = initialize()

av_user(a) = (a.transport_choice == 1)
rh_user(a) = (a.transport_choice == 6)


avcount(model) = sum(model.AVs_time_series)
rhcount(model) = sum(model.RH_trips_time_series)
steps = 500
adata = [(av_user, count), (rh_user, count)]
mdata = [avcount, rhcount]

adf, mdf = run!(model, agent_step!, model_step!, steps; adata, mdata)

function plot_population_timeseries(mdf)
    figure = Figure(resolution = (600, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
    av_population = lines!(ax, mdf.step, mdf.avcount, color = :green)
    rh_population = lines!(ax, mdf.step, mdf.rhcount, color = :blue)
    figure[1, 2] = Legend(figure, [av_population, rh_population], ["AVs in population", "RH trips in population"])
    figure
end

plot_population_timeseries(mdf)













