from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Train the land price and construction cost prediction model
datasets = {
    'Karnataka': pd.read_csv('Karnataka_land_prices.csv'),
    'Delhi': pd.read_csv('Delhi_land_prices.csv'),
    'Maharashtra': pd.read_csv('Maharashtra_land_prices.csv'),
    'Tamil Nadu': pd.read_csv('Tamilnadu_land_prices.csv')
}

# Train the land price and construction cost prediction model
def train_cost_prediction_model(state):
    land_data = datasets[state]  # Select the dataset based on the state
    model = MultiOutputRegressor(LinearRegression())
    X = land_data[['year', 'locality']]  # Features (year and locality)
    y = land_data[['price', 'brick_cost']]  # Multiple target variables

    # Create a column transformer with OneHotEncoder for locality
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), ['locality'])],
        remainder='passthrough'  # Keep other columns as they are
    )

    # Create a pipeline that first transforms the data and then fits the model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', MultiOutputRegressor(LinearRegression()))])

    model.fit(X, y)
    return model



# Function to predict land price and construction costs
def predict_costs(year, state, locality):
    model = train_cost_prediction_model(state)
    input_data = pd.DataFrame({'year': [year], 'locality': [locality]})
    predictions = model.predict(input_data)
    
    # Extract predictions as floats
    land_price = float(predictions[0][0])
    brick_cost = float(predictions[0][1])
    
    return land_price, brick_cost

def format_currency(amount):
    return "₹" + "{:,.0f}".format(amount).replace(",", ",")

@app.route('/')
def landing_page():
    return render_template('home_page.html')


# Function to adjust costs based on year
def adjust_costs_by_year(year, base_cost):
    if year < 2024:
        raise ValueError("Year cannot be less than 2024")
    
    # Define year-based adjustments (example: 5% annual increase)
    year_diff = year - 2024
    
    # Adjustments for different costs
    electric_cost_multiplier = 0.3 * year_diff  #3% increase per year
    steel_cost_multiplier = 0.4 * year_diff  #4% increase per year
    tile_cost_multiplier = 0.2 * year_diff  #2% increase per year
    cement_cost_multiplier = 0.35 * year_diff  #3.5% increase per year
    foundation_cost_multiplier = 0.25 * year_diff  #2.5% increase per year
    wall_construction_cost_multiplier = 0.35 * year_diff

    # Apply the multipliers to adjust the costs
    adjusted_costs = {
        'electric_cost': base_cost['electric_cost'] * electric_cost_multiplier,
        'steel_cost': base_cost['steel_cost'] * steel_cost_multiplier,
        'tile_cost': base_cost['tile_cost'] * tile_cost_multiplier,
        'cement_cost': base_cost['cement_cost'] * cement_cost_multiplier,
        'foundation_cost': base_cost['foundation_cost'] * foundation_cost_multiplier,
        'wall_construction_cost': base_cost['wall_construction_cost'] * wall_construction_cost_multiplier,  # Adjust wall cost
    }
    
    return adjusted_costs

@app.route('/calculate', methods=['GET', 'POST'])
def calculate():
    if request.method == 'POST':
        # Get data from the form
        year = int(request.form.get('year'))
        state = request.form.get('state')
        locality = request.form.get('locality')
        length = float(request.form.get('areaLength'))  # Changed to float for accurate calculations
        breadth = float(request.form.get('areaBreadth'))  # Changed to float for accurate calculations
        num_floors = int(request.form.get('floors'))
        num_bedrooms = int(request.form.get('bedrooms'))
        interior_type = request.form.get('interiorType')
        
        area = length * breadth

        land_price, brick_cost = predict_costs(year, state, locality)

        area_land_price = land_price * area
        
        # Round the land price to the nearest thousand (just like wall construction cost)
        area_land_price = round(area_land_price, -3)

        # Base costs for the year 2024
        base_cost = {
            'electric_cost': 200 * area,
            'steel_cost': 650 * area,
            'tile_cost': area * 100 * num_floors,
            'cement_cost': area * 0.3 * 330 * num_floors,
            'foundation_cost': 250 * area,
            'wall_construction_cost': 400*area
        }
        
        # Adjust costs based on the year
        adjusted_costs = adjust_costs_by_year(year, base_cost)
        
        electric_cost = adjusted_costs['electric_cost']
        steel_cost = adjusted_costs['steel_cost']
        tile_cost = adjusted_costs['tile_cost']
        cement_cost = adjusted_costs['cement_cost']
        foundation_cost = adjusted_costs['foundation_cost']

        # Calculate additional costs based on inputs
        piping_cost = 150 * 25 * (num_bedrooms - 1)
        bathroom_acc = 25000 * (num_bedrooms - 1)

        # Interior cost calculation
        if interior_type == "Minimalistic":
            interior_cost = 500 * area
        elif interior_type == "Modern":
            interior_cost = 800 * area
        elif interior_type == "Luxury":
            interior_cost = 1200 * area
        else:
            interior_cost = 0  # Default case, if no match found

        adjusted_wall_construction_cost = adjusted_costs['wall_construction_cost']
        wall_construction = adjusted_wall_construction_cost
        bathroom_cost = piping_cost + bathroom_acc
        wall_construction -= (wall_construction % 1000)

        total_cost = area_land_price + wall_construction + tile_cost + cement_cost + foundation_cost + steel_cost + electric_cost + piping_cost + bathroom_acc + interior_cost

        # Round the total cost to the nearest thousand
        total_cost = round(total_cost, -3)

        # Calculate estimated price range (±10%)
        lower_bound = total_cost * 0.95
        upper_bound = total_cost * 1.05

        # Round the estimated price range to the nearest thousand
        lower_bound = round(lower_bound, -3)
        upper_bound = round(upper_bound, -3)

        # Format amounts for currency display
        formatted_area_land_price = format_currency(area_land_price)
        formatted_total_cost = format_currency(total_cost)
        formatted_lower_bound = format_currency(lower_bound)
        formatted_upper_bound = format_currency(upper_bound)

        return render_template('output.html', 
                              total_cost=formatted_total_cost, 
                              area_land_price=formatted_area_land_price, 
                              wall_construction=format_currency(wall_construction), 
                              tile_cost=format_currency(tile_cost), 
                              cement_cost=format_currency(cement_cost), 
                              foundation_cost=format_currency(foundation_cost), 
                              steel_cost=format_currency(steel_cost), 
                              electric_cost=format_currency(electric_cost), 
                              bathroom_cost=format_currency(bathroom_cost), 
                              interior_cost=format_currency(interior_cost),
                              lower_bound=formatted_lower_bound, 
                              upper_bound=formatted_upper_bound)

    return render_template('index.html')



@app.route('/output', methods=['GET', 'POST'])
def output():
    # Since the result is rendered by the `calculate` function, no additional logic is needed here
    return redirect('/')

    
    return render_template('output.html', state=state, locality=locality)

if __name__ == '__main__':
    app.run(debug=True)