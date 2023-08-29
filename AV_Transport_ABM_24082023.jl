using Agents
using Random
using CSV
using Distributions
using DataFrames
using Plots
using GLMakie

space = GridSpace((20, 20); periodic = false)

abstract type AllAgents end



Random.seed!(1234)
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

@agent TransportAgent <: AllAgents GridAgent{2} begin
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
    rh_fee_applied::Bool

    # Public transport layer 
    near_public_transport::Bool

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

    # PUBLIC TRANSPORT AGENTS 
    is_pt_agent::Bool
    pt_coverage_radius::Int 
    fee::Int 

    # PROMOTION AGENT 
    is_promotion_agent::Bool
    promotion_coverage_radius::Int

end



function init_public_transport_agents!(model)
    for _ in 1:model.num_public_transport_agents
        add_agent!(PublicTransportAgent, model, 3, 0)
    end
end

function add_promotion_agents!(model)
    for _ in 1:model.num_promotion_agents
        add_agent!(PromotionAgent, model, 2)
    end
end

function initialize(; total_agents = 250, griddims = (20, 20), private_AV_cost = 20000, rh_trip_cost = 10, seed = 100, av_threshold_model = 5.0, rh_threshold_model = 5.0, AVs = 0, RH_trips = 0, AVs_time_series = [0], # Starting with 0 AVs
    RH_trips_time_series = [0], rh_fee_applied = false, num_public_transport_agents = 100, num_promotion_agents = 50)
    rng = MersenneTwister(seed)
    space = GridSpace(griddims, periodic = false)
    properties = Dict(:private_AV_cost => private_AV_cost, :rh_trip_cost => rh_trip_cost, :tick => 1, :av_threshold_model => av_threshold_model, :rh_threshold_model => rh_threshold_model, :AVs => 0, :RH_trips => 0, :AVs_time_series => [0], :RH_trips_time_series => [0], :total_agents => total_agents, :rh_fee_applied => false, :num_public_transport_agents => 100, :num_promotion_agents => 50)
    model = ABM(Union{TransportAgent, PublicTransportAgent, PromotionAgent}, space; properties = properties, rng, scheduler = Schedulers.Randomly(), warn = false
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
        rh_fee_applied = false

        #Public transport layer
        near_public_transport = false

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

        # PUBLIC TRANSPORT AGENTS 
        is_pt_agent = false
        pt_coverage_radius = 0 
        fee = 0

        #  PROMOTION AGENT 
        is_promotion_agent = false 
        promotion_coverage_radius = 0


        add_agent!(TransportAgent, model, age, gender, education, employment, income, original_transport_type, transport_type, transport_choice, av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold, rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, rh_fee_applied, near_public_transport, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg, physical_health_layer, sedentary_behaviour, is_pt_agent, pt_coverage_radius, fee, is_promotion_agent, promotion_coverage_radius)
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
        rh_fee_applied = false

        # Public transport layer
        near_public_transport = false

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

        # PUBLIC TRANSPORT AGENTS 
        is_pt_agent = false
        pt_coverage_radius = 0 
        fee = 0

        #  PROMOTION AGENT 
        is_promotion_agent = false 
        promotion_coverage_radius = 0


        add_agent!(TransportAgent, model, age, gender, education, employment, income, original_transport_type, transport_type, transport_choice, av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold, rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, rh_fee_applied, near_public_transport, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg, physical_health_layer, sedentary_behaviour, is_pt_agent, pt_coverage_radius, fee, is_promotion_agent, promotion_coverage_radius)
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
        rh_fee_applied = false

        #Public transport layer 
        near_public_transport = false

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

        # PUBLIC TRANSPORT AGENTS 
        is_pt_agent = false
        pt_coverage_radius = 0 
        fee = 0

        #  PROMOTION AGENT 
        is_promotion_agent = false 
        promotion_coverage_radius = 0


        add_agent!(TransportAgent, model, age, gender, education, employment, income, original_transport_type, transport_type, transport_choice, av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold, rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, rh_fee_applied, near_public_transport, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg, physical_health_layer, sedentary_behaviour, is_pt_agent, pt_coverage_radius, fee, is_promotion_agent, promotion_coverage_radius)
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
        rh_fee_applied = false

        #Public transport layer
        near_public_transport = false

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


        # PUBLIC TRANSPORT AGENTS 
        is_pt_agent = false
        pt_coverage_radius = 0 
        fee = 0

        #  PROMOTION AGENT 
        is_promotion_agent = false 
        promotion_coverage_radius = 0


        add_agent!(TransportAgent, model, age, gender, education, employment, income, original_transport_type, transport_type, transport_choice, av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold, rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, rh_fee_applied, near_public_transport, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg, physical_health_layer, sedentary_behaviour, is_pt_agent, pt_coverage_radius, fee, is_promotion_agent, promotion_coverage_radius)
    end 

    # PUBLIC TRANSPORT AGENTS 

    for _ in 1:model.num_public_transport_agents
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
        rh_fee_applied = false

        #Public transport layer
        near_public_transport = false

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


        # PUBLIC TRANSPORT AGENTS 
        is_pt_agent = true
        pt_coverage_radius = 3 
        fee = 0

        #  PROMOTION AGENT 
        is_promotion_agent = false 
        promotion_coverage_radius = 0


        add_agent!(TransportAgent, model, age, gender, education, employment, income, original_transport_type, transport_type, transport_choice, av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold, rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, rh_fee_applied, near_public_transport, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg, physical_health_layer, sedentary_behaviour, is_pt_agent, pt_coverage_radius, fee, is_promotion_agent, promotion_coverage_radius)
    end 

    # PROMOTION AGENTS 

    for _ in 1:model.num_public_transport_agents
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
        rh_fee_applied = false

        #Public transport layer
        near_public_transport = false

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


        # PUBLIC TRANSPORT AGENTS 
        is_pt_agent = false
        pt_coverage_radius = 0 
        fee = 0

        #  PROMOTION AGENT 
        is_promotion_agent = true 
        promotion_coverage_radius = 2


        add_agent!(TransportAgent, model, age, gender, education, employment, income, original_transport_type, transport_type, transport_choice, av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold, rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, rh_fee_applied, near_public_transport, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg, physical_health_layer, sedentary_behaviour, is_pt_agent, pt_coverage_radius, fee, is_promotion_agent, promotion_coverage_radius)
    end 

    return model 
end



model = initialize()

function agent_step!(agent, model)
    update_near_public_transport(agent, model)
    consolidated_transport_decision!(agent, model)
    agent_health!(agent, model)
    # apply_rebate_after_purchase!(agent, model)
end

function model_step!(model)
    model.tick += 1

    if model.tick <= 25
        model.av_threshold_model = 4
        model.rh_threshold_model = 4
    
    elseif model.tick > 20 && model.tick <= 50
        model.av_threshold_model = 3
        model.rh_threshold_model = 3

    elseif model.tick > 50 && model.tick <= 75
        model.av_threshold_model = 2
        model.rh_threshold_model = 2
    
    elseif model.tick > 75 && model.tick <= 90
        model.av_threshold_model = 1.5
        model.rh_threshold_model = 1.5

    else 
        model.av_threshold_model = 1
        model.rh_threshold_model = 1
    end

end

# Assign whether the agents are near public transport

function update_near_public_transport(agent, model)
    for pt_agent in nearby_agents(agent, model, 3)
        if isa(pt_agent, PublicTransportAgent)
            agent.near_public_transport = true
            return
        end
    end
    agent.near_public_transport = false
end



# AV DECISION-MAKING USING THE THEORY OF PLANNED BEHAVIOUR # 
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

# RIDE-HAIL DECISION-MAKING USING THE THEORY OF PLANNED BEHAVIOUR # 

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



# POLICIES: ASSIGNING FEES FOR SHORT RIDE-HAIL TRIPS AND TRIPS WHERE PUBLIC TRANSPORT IS NEARBY 
# If the randomly generated trip for a ride-hail is a short distance, then there will be a 50% increase in price. 
function assign_rh_trip_cost(agent_trip_distance, agent, model)

    trip_cost = model.rh_trip_cost # start with the default cost

    # FEES FOR SHORT TRIPS 
    short_trip_threshold = 5 # You can adjust this value.
    short_trip_fee_amount = 5 # Example fee amount for short trips.

    if agent_trip_distance <= short_trip_threshold
        trip_cost += short_trip_fee_amount
    end

    # FEES FOR TRIPS WHERE PUBLIC TRANSPORT IS NEARBY 
    nearby_public_transport_fee = 20

    if agent.near_public_transport
        trip_cost += nearby_public_transport_fee
    end

    # print("Price: $trip_cost")
    return trip_cost
end

# POLICY: VISIBLE REBATE FOR AGENTS
# This rebate changes the price of the AV so that agents can see it decrease, essentially appearing as a 'discount' for private AVs. 

AV_rebate_full_amount = 1000

function apply_rebate!(agent, model, rebate_amount)

    private_av_price = model.private_AV_cost - rebate_amount # Reduce the cost of private AV by the rebate amount

    return private_av_price
end


function consolidated_transport_decision!(agent, model)
    # Calculate AV Decision
    av_attitudes = agent.av_attitudes
    av_control_behaviour = agent.av_cb_pos - 2*agent.av_cb_neg
#     # calculating social_norms
    av_subjective_norms = 0
    av_num_neighbors = 0
    for av_neighbor in nearby_agents(agent, model)
        agent_av_subjective_norms = av_subjective_norms + av_neighbor.av_attitudes
        av_num_neighbors = av_num_neighbors + 1
    end
    
    #Taking the average
    av_subjective_norms = av_num_neighbors==0 ? 0 : av_subjective_norms / av_num_neighbors    

    av_descriptive_norms = model.AVs / model.total_agents
    # COMMENT THIS OUT TO TURN OFF THE REBATE POLICY
    av_facil_conditions = [agent.income > apply_rebate!(agent, model, AV_rebate_full_amount)]
    av_decision = AV_TPB(av_attitudes, av_control_behaviour, av_subjective_norms, av_descriptive_norms, av_facil_conditions, model.av_threshold_model)

    # Calculate Ride-Hail Decision
    rh_attitudes = agent.rh_attitudes
    rh_control_behaviour = agent.rh_cb_pos - 2*agent.rh_cb_neg
    rh_descriptive_norms = model.RH_trips / model.total_agents
    # calculating social_norms
    rh_subjective_norms = 0
    rh_num_neighbors = 0
    for rh_neighbor in nearby_agents(agent, model)
        agent_rh_subjective_norms = rh_subjective_norms + rh_neighbor.rh_attitudes
        rh_num_neighbors = rh_num_neighbors + 1
    end
    
    #Taking the average
    rh_subjective_norms = rh_num_neighbors==0 ? 0 : rh_subjective_norms / rh_num_neighbors   
    agent_trip_distance = rand(1:20)
    # IF FEES FOR SHORT TRIPS AND FOR NEARBY PUBLIC TRANSPORT ARE BEING IMPLEMENTED, ACTIVATE CODE BELOW:
    rh_facil_conditions = [(agent.income*0.0005) > assign_rh_trip_cost(agent_trip_distance, agent, model)] 
    # IF FEES FOR SHORT TRIPS POLICY IS NOT BEING IMPLEMENTED, ACTIVATE CODE BELOW: 
    # rh_facil_conditions = [(agent.income*0.001) > model.rh_trip_cost]
    rh_decision = RH_TPB(rh_attitudes, rh_control_behaviour, rh_subjective_norms, rh_descriptive_norms, rh_facil_conditions, model.rh_threshold_model)
    
    # Final Decision
    # This is an example of a decision hierarchy - you can adjust as required.
    if av_decision 
        agent.transport_choice = 1
        push!(model.AVs_time_series, model.AVs_time_series[end] + 1)
    elseif rh_decision
        agent.transport_choice = 6
        push!(model.RH_trips_time_series, model.RH_trips_time_series[end] + 1)
    else
        agent.transport_choice = agent.original_transport_type
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


# POLICY: 10% REBATE ON THE PURCHASE PRICE 
# Note: the results from this do not change anything, as it does not change the perceived purchase price. Option 2 will address this. 

const AV_REBATE = 0.1 # 10% rebate for using AV

function apply_rebate_after_purchase!(agent, model)
    if agent.transport_choice == 1 # If choice is AV
        rebate_amount = model.private_AV_cost * AV_REBATE
        agent.income += rebate_amount # Apply the rebate to the income
    end
end



adata = [:pos, :transport_type, :transport_choice, :age, :income, :original_transport_type, :av_attitudes, :av_social_norms, :av_control_factors, :av_behavioural_intention, :rh_attitudes, :rh_social_norms, :rh_control_factors, :rh_behavioural_intention, :impulsivity, :physical_health_layer, :sedentary_behaviour]
data, _ = run!(model, agent_step!, model_step!, 365; adata)



using CairoMakie # using a different plotting backend that enables interactive plots
using Statistics


model = initialize()


av_user(a) = (a.transport_choice == 1)
rh_user(a) = (a.transport_choice == 6)
car_user(a) = (a.transport_choice == 2)
pt_user(a) = (a.transport_choice == 3)
cyclist(a) = (a.transport_choice == 4)
walker(a) = (a.transport_choice == 5)

print(model.AVs_time_series)
print(model.RH_trips_time_series)


avcount(model) = sum(model.AVs_time_series)
rhcount(model) = sum(model.RH_trips_time_series)
steps = 500
adata = [(av_user, count), (rh_user, count), (car_user, count), (pt_user, count), (cyclist, count), (walker, count)]
mdata = [avcount, rhcount]

adf, mdf = run!(model, agent_step!, model_step!, steps; adata, mdata)

function plot_population_timeseries(adf)
    figure = Figure(resolution = (600, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
    av_agents = lines!(ax, adf.step, adf.count_av_user, color = :blue)
    rh_agents = lines!(ax, adf.step, adf.count_rh_user, color = :green)
    car_agents = lines!(ax, adf.step, adf.count_car_user, color = :purple)
    pt_agents = lines!(ax, adf.step, adf.count_pt_user, color = :orange)
    cyclist_agents = lines!(ax, adf.step, adf.count_cyclist, color = :red)
    walker_agents = lines!(ax, adf.step, adf.count_walker, color = :pink)
    # av_population = lines!(ax, mdf.step, mdf.avcount, color = :green)
    # rh_population = lines!(ax, mdf.step, mdf.rhcount, color = :blue)
    figure[1, 2] = Legend(figure, [av_agents, rh_agents, car_agents, pt_agents, cyclist_agents, walker_agents], ["AVs", "RH users", "Car users", "Public transport", "Cyclists", "Walkers"])
    figure
end


# CSV.write("C:/Users/godicb/OneDrive - The University of Melbourne/Documents/Julia/AV_Transport_ABM/output8_25082023.csv" ,data)


plot_population_timeseries(adf)

# OLD DECISION FUNCTIONS 


# function av_decision!(agent, model)
#     attitudes = agent.av_attitudes
#     control_behaviour = agent.av_cb_pos - 2*agent.av_cb_neg
#     # calculating social_norms
#     subjective_norms = 0
#     num_neighbors = 0
#     for neighbor in nearby_agents(agent, model)
#         subjective_norms = subjective_norms + neighbor.av_attitudes
#         num_neighbors = num_neighbors + 1
#     end
    
#     #Taking the average
#     subjective_norms = num_neighbors==0 ? 0 : subjective_norms / num_neighbors

#     #Taking the average
#     descriptive_norms = model.AVs / model.total_agents


#     # REBATE POLICY OPTION 2:
#     # If a rebate policy with the full amount is being applied, included, ensure that the below code is activated:
#     apply_rebate!(agent, model, AV_rebate_full_amount)

#     facil_conditions = [agent.income > model.private_AV_cost]

#     AV_decision = AV_TPB(attitudes, control_behaviour, subjective_norms, descriptive_norms, facil_conditions, model.av_threshold_model)
    
#     if AV_decision 
#         push!(model.AVs_time_series, model.AVs_time_series[end] + 1)
#     else
#         push!(model.AVs_time_series, model.AVs_time_series[end])
#     end

#     if AV_decision
#         agent.transport_choice = 1
#     end

# end



# function rh_decision!(agent, model)
#     attitudes = agent.rh_attitudes
#     control_behaviour = agent.rh_cb_pos - 2*agent.rh_cb_neg
#     # calculating social_norms
#     subjective_norms = 0
#     num_neighbors = 0
#     for neighbor in nearby_agents(agent, model)
#         subjective_norms = subjective_norms + neighbor.rh_attitudes
#         num_neighbors = num_neighbors + 1
#     end
    
#     # Taking the average
#     subjective_norms = num_neighbors == 0 ? 0 : subjective_norms / num_neighbors

#     # Taking the average
#     descriptive_norms = model.RH_trips / model.total_agents
    
#     agent_trip_distance = rand(1:20)

#     print("Trip distance: $agent_trip_distance")

#     # SHORT TRIP FEE POLICY IMPLEMENTED - ACTIVATE CODE BELOW #

#     if agent_trip_distance <= 5
#         facil_conditions = [(agent.income*0.001) > assign_rh_trip_cost(agent_trip_distance, agent, model)]
#     else
#         facil_conditions = [(agent.income*0.001) > model.rh_trip_cost]
#     end

#     # SHORT TRIP FEE POLICY NOT IMPLEMENTED - ACTIVATE CODE BELOW #

#     #facil_conditions = [(agent.income*0.001) > model.rh_trip_cost]

#     RH_decision = RH_TPB(attitudes, control_behaviour, subjective_norms, descriptive_norms, facil_conditions, model.rh_threshold_model)

#     if RH_decision 
#         push!(model.RH_trips_time_series, model.RH_trips_time_series[end] + 1)
#     else
#         push!(model.RH_trips_time_series, model.RH_trips_time_series[end])
#     end

#     if RH_decision
#         agent.transport_choice = 6
#     end

# end











