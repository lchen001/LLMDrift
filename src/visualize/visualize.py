import plotly.offline as pyo
import plotly.io as py
import plotly.graph_objects as go

class Visualize(object):
    def __init__(self,root_path='../../figure/full/'):
        self.root_path=root_path
        return 
    
    def plot_bars(self,data,
                  errors_bar,
                 colors=['#1f77b4','#ff7f0e'],
                  fontsize=25,
                  yrange=[0,1],
                  name_map={'openaichat/gpt-4-0314':'March 2023',"openaichat/gpt-4-0613":"June 2023"},
                 no_text=False,
                 ):
        # Select the desired data
        names = [key1 for key1 in name_map]
        selected_data = data[names].rename(index=name_map)
        
        errors = errors_bar[names].rename(index=name_map)
        percentage_values = selected_data.values * 100
        # Create the bar plot
        if(no_text==True):
            fig = go.Figure(data=[go.Bar(
            x=selected_data.index,
            y=selected_data.values,
            text=[f'{val:.1f}' for val in selected_data.values],  # display values as percentages
                
            textposition='auto',
            marker_color=colors,  # change this to the colors you want
            
            error_y=dict(type='data', array=errors.values, visible=True),  # add error bars
        )])
        else:
            fig = go.Figure(data=[go.Bar(
                x=selected_data.index,
            y=selected_data.values,
            text=[f'{val:.1f}%' for val in percentage_values],  # display values as percentages
            textposition='auto',
            marker_color=colors,  # change this to the colors you want
            
            error_y=dict(type='data', array=errors.values, visible=True),  # add error bars
        )])

        fig.update_layout(
            autosize=True,
            margin=go.layout.Margin(l=150, r=0, b=0, t=0),
            width=700,  # set figure width here
            height=500,  # set figure height here
            #title_text='Performance of Selected Models',
            font=dict(
                size=fontsize,  # change this to the font size you want
                family="Arial"  # set the font to Arial

            ),
            
            #yaxis=dict(range=yrange),  # manually set the range of the y-axis
            yaxis=dict(range=[0, max(max(selected_data.values),yrange[1])]),  # adjust y-axis range to accommodate error bars

            #yaxis=dict(range=[0, max(selected_data.values + errors.values)]),  # adjust y-axis range to accommodate error bars
            #yaxis_title=selected_data.name,  # add y-axis label here

        )
        fig.show()   
        self.fig = fig
        
    def plot_bar(self,score,
                  errors_bar,
                 colors=['#9467bd','#1f77b4','#ff7f0e'],
                  fontsize=25,
                  yrange=[0,1],
                  name_map={'openaichat/gpt-4-0314':'March 2023',"openaichat/gpt-4-0613":"June 2023"},
                 ):
        # Select the desired data
        names = ['']
        selected_data = [score]
        
        errors = [errors_bar]
        #selected_data = data.loc[['openaichat/gpt-4-0314', 'openaichat/gpt-4-0613']]
        #print("selected_data",selected_data)
        percentage_values = [temp*100 for temp in selected_data] 

        # Create the bar plot
        fig = go.Figure(data=[go.Bar(
            x=names,
            y=selected_data,
        
    
            text=[f'{y:.1f}%' for y in percentage_values],  # format y_data as percentages
            textposition='auto',
            marker_color=colors,  # change this to the colors you want
            
            error_y=dict(type='data', array=errors, visible=True),  # add error bars
        )])
        
        errors_max = [selected_data[i]+errors[i] for i in range(len(selected_data))]
        #'''
        fig.update_layout(
            autosize=True,
            margin=go.layout.Margin(l=150, r=0, b=0, t=0),
            width=350,  # set figure width here
            height=500,  # set figure height here
            #title_text='Performance of Selected Models',
            font=dict(
                size=fontsize,  # change this to the font size you want
                family="Arial"  # set the font to Arial

            ),
            
             yaxis=dict(range=yrange),  # manually set the range of the y-axis
            #yaxis=dict(range=[0, max(errors_max)]),  # adjust y-axis range to accommodate error bars
            yaxis_title="",  # add y-axis label here

        )
        
        #print("error range",[0, max(selected_data + errors)],errors)
        #print("selected_data",selected_data,errors,errors+selected_data)
        #'''

        fig.show()   
        self.fig = fig        
        
    def save_figure(self, filename):
        py.write_image(self.fig, self.root_path+filename)