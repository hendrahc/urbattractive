class ScoresController < ApplicationController
  def new
    session[:lgt] = session[:images].length
    if session[:img_num].to_i < session[:images].length
      @score = Score.new
      #set image to be shown
      @img = Image.find((session[:images][(session[:img_num].to_i)].to_i))
    else
       redirect_to end_path
    end 
  end

  def create
  #save score value to database (img id, user id, score)
    @score = Score.new(score_params)
    if @score.save
      
      if session[:img_num].to_i < session[:images].length

          redirect_to checkimage_path
          #redirect_to show_path
      else
          redirect_to end_path
      end 
      
    else
      render '/new'
    end
  end

  def contentcheck 
    if session[:checkimage] == 1
          redirect_to content_path
    else
        session[:img_num] = session[:img_num].to_i + 1 #increment image id 
        if session[:img_num].to_i < session[:images].length
          redirect_to show_path
        else
          redirect_to end_path
        end 
      end 
  end

  def checkimage
      if (session[:img_num]).to_i == 5 || (session[:img_num]).to_i == 19
          session[:checkimage] = 1
      else 
          session[:checkimage] = 0
      end
      redirect_to contentcheck_path
  end

private

  def score_params
    params.require(:score).permit(:score, :img_id, :user_id, :scale, :recognizability, :lgt)
  end

end
